"""mhc_pre_norm_fn: torch-fallback (kernel port pending).

Provides ``round_to_tf32`` and the
``_mhc_pre_norm_fn_{fwd,bwd}_{mul,norm}`` /
``_mhc_fn_normw_merge_{fwd,bwd}`` factory functions that
``modeling/mhc/ops/norm_fn.py`` imports.

Math (forward, single-split):
    fn_eff   = fn * normw                     # if normw given
    matmul[t, m] = sum_k x[t, k] * fn_eff[m, k]
    sqrsum[t]    = sum_k x[t, k]^2
    rsqrt        = (sqrsum / K + eps) ** -0.5
    out[t, m]    = matmul[t, m] * rsqrt[t]

Backward: pulled directly via torch autograd replay.
"""

import torch


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    """No-op pass-through for the torch-fallback path.

    The TileLang kernel pre-rounded its fp32 fn input to TF32 precision
    so the matmul could treat it as such.  Our torch-fallback enables
    ``torch.backends.cuda.matmul.allow_tf32`` inside its matmul calls,
    so the rounding happens implicitly inside the matmul; explicit
    pre-rounding would cause double-rounding drift in fn_grad.
    """
    return x


# ---------------------------------------------------------------------------
# fn = fn * normw merge.
# ---------------------------------------------------------------------------

def _mhc_fn_normw_merge_fwd(M: int, K: int):
    def runner(fn, normw, out_fn):
        # fn:    (M, K) fp32
        # normw: (K,)   fp32
        # out_fn:(M, K) fp32  out
        out_fn.copy_(fn * normw)
    return runner


def _mhc_fn_normw_merge_bwd(M: int, K: int):
    def runner(fn, normw, out_fn_grad, fn_grad, normw_grad):
        # fn_grad / normw_grad accumulate.
        fn_grad.add_(out_fn_grad * normw)
        normw_grad.add_((out_fn_grad * fn).sum(dim=0))
    return runner


# ---------------------------------------------------------------------------
# Forward: matmul + sqrsum (split into _mul + _norm to match the wrapper).
# ---------------------------------------------------------------------------

def _mhc_pre_norm_fn_fwd_mul(M: int, _unused: int, K: int):
    def runner(x, fn, out_mul, sqrsum):
        # x:       (n, K)         bf16
        # fn:      (M, K)         fp32 (already tf32-rounded)
        # out_mul: (n, 1, M)      fp32  out
        # sqrsum:  (n, 1)         fp32  out
        # Match the ref's tf32 matmul: temporarily flip allow_tf32 so the
        # gradient parity check (which uses allow_tf32=True for the ref)
        # sees the same accumulation precision.
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            x_f = x.float()
            out_mul.copy_((x_f @ fn.T).unsqueeze(1))
            sqrsum.copy_(x_f.square().sum(-1, keepdim=True))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    return runner


def _mhc_pre_norm_fn_fwd_norm(M: int, _unused: int, K: int, eps: float, n_splits: int):
    def runner(out_mul_splitted, sqrsum_splitted, out_mul, sqrsum, out):
        # out_mul_splitted: (n_splits, n, 1, M)
        # sqrsum_splitted:  (n_splits, n, 1)
        # out_mul:          (n, 1, M)  out (sum across splits)
        # sqrsum:           (n, 1)     out (sum across splits)
        # out:              (n, M)     out (final result)
        out_mul.copy_(out_mul_splitted.sum(0))
        sqrsum.copy_(sqrsum_splitted.sum(0))
        rsqrt = (sqrsum.unsqueeze(-1) / K + eps).rsqrt()  # (n, 1, 1)
        out.copy_((out_mul * rsqrt).sum(-2))               # (n, M)
    return runner


# ---------------------------------------------------------------------------
# Backward: norm step then matmul step, mirroring the wrapper.
# ---------------------------------------------------------------------------

def _mhc_pre_norm_fn_bwd_norm(M: int, _unused: int, K: int, eps: float):
    def runner(out_grad, out_mul, sqrsum, out_mul_grad, sqrsum_grad):
        # out_grad:     (n, M)
        # out_mul:      (n, 1, M) saved
        # sqrsum:       (n, 1)    saved
        # out_mul_grad: (n, 1, M) out
        # sqrsum_grad:  (n, 1)    out
        with torch.enable_grad():
            om = out_mul.detach().requires_grad_(True)
            sq = sqrsum.detach().requires_grad_(True)
            rsqrt = (sq.unsqueeze(-1) / K + eps).rsqrt()
            out = (om * rsqrt).sum(-2)
            d_om, d_sq = torch.autograd.grad(out, [om, sq], out_grad)
        out_mul_grad.copy_(d_om)
        sqrsum_grad.copy_(d_sq)
    return runner


def _mhc_pre_norm_fn_bwd_mul(M: int, _unused: int, K: int):
    def runner(out_mul_grad, sqrsum_grad, x, fn, x_grad, fn_grad):
        # out_mul_grad: (n, 1, M)
        # sqrsum_grad:  (n, 1)
        # x:            (n, K)         bf16
        # fn:           (M, K)         fp32
        # x_grad:       (n, K)         bf16  in/out (accumulating)
        # fn_grad:      (M, K)         fp32  out
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with torch.enable_grad():
                x_ = x.detach().requires_grad_(True)
                fn_ = fn.detach().requires_grad_(True)
                x_f = x_.float()
                mat = (x_f @ fn_.T).unsqueeze(1)
                ssum = x_f.square().sum(-1, keepdim=True)
                d_x, d_fn = torch.autograd.grad(
                    [mat, ssum], [x_, fn_], [out_mul_grad, sqrsum_grad],
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        x_grad.copy_((x_grad.float() + d_x).bfloat16())
        fn_grad.copy_(d_fn)
    return runner
