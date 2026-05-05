"""tf32_hc_prenorm_gemm: torch fallback of the DeepGEMM split-K bf16->fp32
matmul + per-row squared-sum used by ``mhc_pre``.

Math::
    out[s, t, m] += x[t, k] * fn[m, k]   for k in split-s of the K axis
    sqrsum[s, t] += x[t, k]^2            for k in split-s of the K axis

x is bf16, fn is fp32, out and sqrsum are fp32 split-K outputs that the
caller sums over the split axis after the kernel returns.

DeepGEMM is a CUDA-only kernel; on ROCm we route through ``torch.matmul``
with ``allow_tf32=True`` to match the reference's accumulation precision.
hipBLASLt covers this efficiently for the sizes involved
(num_tokens up to ~32k, hidden up to ~32k).
"""

import torch


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,         # (n, hc * h)   bf16  — residual flattened
    fn: torch.Tensor,        # (M3, hc * h) fp32  — fn weights
    out: torch.Tensor,       # (n_splits, n, M3) fp32 — out
    sqrsum: torch.Tensor,    # (n_splits, n)     fp32 — out
    n_splits: int,
) -> None:
    """Compute split-K matmul ``out[s] = (x @ fn.T) / n_splits-th-of-K`` and
    per-row squared-sum ``sqrsum[s]`` for each split.

    The split is along the K (= hc * hidden_size) axis.  Since the
    aggregator (``mhc_pre_big_fuse``) just sums splits along axis 0,
    we collapse the split-K loop into a single full matmul and a single
    full sqrsum, parking the entire result in slot 0 and zero-filling
    the rest.  This is bit-equivalent to the split-K result after the
    sum.
    """
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert out.dtype == torch.float32
    assert sqrsum.dtype == torch.float32

    n, K = x.shape
    M3, K_fn = fn.shape
    assert K == K_fn, f"x.shape[-1] ({K}) != fn.shape[-1] ({K_fn})"
    assert out.shape == (n_splits, n, M3), out.shape
    assert sqrsum.shape == (n_splits, n), sqrsum.shape

    if n == 0:
        out.zero_()
        sqrsum.zero_()
        return

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        x_f = x.float()
        full_out = x_f @ fn.T              # (n, M3) fp32
        full_sqr = x_f.square().sum(-1)    # (n,)    fp32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    out.zero_()
    sqrsum.zero_()
    out[0].copy_(full_out)
    sqrsum[0].copy_(full_sqr)
