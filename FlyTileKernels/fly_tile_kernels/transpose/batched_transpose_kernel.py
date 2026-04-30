"""batched_transpose / transpose: element-level transpose.

Naive 1-thread-per-element implementation: each thread reads one element
from ``x[b, m, n]`` and writes it to ``out[b, n, m]``.  Correct (bit-exact)
for arbitrary contiguous input, and tolerates the strided-leading-dim
``twice_stride(...)`` shape used by ``test_transpose`` because the source
tensor's stride information is preserved by ``make_buffer_tensor``.

Supported dtypes: bfloat16, float32.  fp8 (e4m3) is not supported here —
the minimum BufferCopy width is 16 bits, so single-byte transposed stores
would overlap between adjacent threads.  An LDS-staged pair-and-pack
implementation is the natural follow-up; the worked-example skeleton in
``references/worked_examples/batched_transpose.md`` is the starting point.

API notes (this FlyDSL build, gfx950): see
``moe/normalize_weight_kernel.py`` for the underlying patterns
(``_dyn(...)`` for dynamic-if; per-element BufferCopy load/store).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec
from flydsl._mlir import ir

from fly_tile_kernels._stub import not_yet_ported


NUM_THREADS = 128


def _dyn(x):
    return x


_DTYPE_INFO = {
    torch.bfloat16: (fx.BFloat16, fx.rocdl.BufferCopy16b),
    torch.float32: (fx.Float32, fx.rocdl.BufferCopy32b),
}


def get_batched_transpose_kernel(dtype: torch.dtype):
    if dtype not in _DTYPE_INFO:
        raise NotImplementedError(
            f"batched_transpose: dtype {dtype} not supported by this naive port; "
            f"supported: {{bfloat16, float32}}."
        )
    fx_dtype, atom_factory = _DTYPE_INFO[dtype]

    @flyc.kernel
    def transpose_kernel(
        x:        fx.Tensor,    # (B, M, N) of dtype, possibly with strided leading dim
        out:      fx.Tensor,    # (B, N, M) of dtype, contiguous
        n_rows:   fx.Int32,     # M
        n_cols:   fx.Int32,     # N
    ):
        b = fx.block_idx.x
        m = fx.block_idx.y
        bid_z = fx.block_idx.z
        tid = fx.thread_idx.x
        n = bid_z * NUM_THREADS + tid

        x_buf = fx.rocdl.make_buffer_tensor(x)
        o_buf = fx.rocdl.make_buffer_tensor(out)

        elem_ty = fx.MemRefType.get(
            fx_dtype.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom = fx.make_copy_atom(atom_factory(), fx_dtype)

        if _dyn(n < n_cols):
            # Load x[b, m, n].
            x_row = fx.slice(x_buf, (b, m, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
            reg = fx.memref_alloca(elem_ty, s_lay)
            fx.copy_atom_call(copy_atom, fx.slice(x_row_div, (None, n)), reg)
            # Store out[b, n, m].
            o_row = fx.slice(o_buf, (b, n, None))
            o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
            fx.copy_atom_call(copy_atom, reg, fx.slice(o_row_div, (None, m)))

    @flyc.jit
    def launch(
        x:        fx.Tensor,
        out:      fx.Tensor,
        n_rows:   fx.Int32,
        n_cols:   fx.Int32,
        n_batches: fx.Int32,
        stream:   fx.Stream = fx.Stream(None),
    ):
        gz = (n_cols + NUM_THREADS - 1) // NUM_THREADS
        transpose_kernel(x, out, n_rows, n_cols).launch(
            grid=(n_batches, n_rows, gz), block=(NUM_THREADS, 1, 1), stream=stream,
        )

    return launch


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D tensor.

    Args:
        x: Input 2D tensor (M, N) with both dims divisible by 64.

    Returns:
        Transposed tensor of shape (N, M).
    """
    if x.dtype == torch.float8_e4m3fn:
        not_yet_ported(
            "transpose",
            "fp8 transpose needs LDS pair-and-pack to avoid sub-16b stores; see batched_transpose.md",
        )
    x = x.unsqueeze(0)
    out = batched_transpose(x)
    return out.squeeze(0)


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a batched 3D tensor.

    Args:
        x: Input 3D tensor (B, M, N) with M, N divisible by 64.

    Returns:
        Transposed tensor of shape (B, N, M).
    """
    if x.dtype == torch.float8_e4m3fn:
        not_yet_ported(
            "batched_transpose",
            "fp8 transpose needs LDS pair-and-pack to avoid sub-16b stores; see batched_transpose.md",
        )

    assert x.dim() == 3
    num_batches, shape_x, shape_y = x.shape
    assert shape_x % 64 == 0 and shape_y % 64 == 0
    assert x.stride(-1) == 1

    out = torch.empty((num_batches, shape_y, shape_x), dtype=x.dtype, device="cuda")
    if num_batches == 0 or shape_x == 0 or shape_y == 0:
        return out

    kernel = get_batched_transpose_kernel(x.dtype)
    kernel(x, out, shape_x, shape_y, num_batches)
    return out
