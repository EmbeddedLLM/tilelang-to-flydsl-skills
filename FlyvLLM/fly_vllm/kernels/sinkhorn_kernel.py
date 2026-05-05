"""sinkhorn: iterative row + column normalisation of small per-token mhc x mhc
matrices.

Forward (per token's (mhc, mhc) matrix):
    x = softmax(x, dim=-1)
    x = x + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)        # column normalise
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)    # row normalise
        x = x / (x.sum(dim=-2, keepdim=True) + eps)    # column normalise

One thread per token; full (mhc, mhc) state held in registers.  ``mhc=4``
⇒ 16 fp32 registers per thread.

Mirrors the inner sinkhorn loop of ``mhc_pre_big_fuse_tilelang`` from
``vllm.model_executor.layers.mhc``.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, math as fxmath
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def _build_sinkhorn_kernel(hidden_size: int, repeat: int, eps: float):
    H = hidden_size

    @flyc.kernel
    def kernel(
        x_in:       fx.Tensor,   # (n, H, H) fp32
        x_out:      fx.Tensor,   # (n, H, H) fp32
        num_tokens: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        in_buf = fx.rocdl.make_buffer_tensor(x_in)
        out_buf = fx.rocdl.make_buffer_tensor(x_out)

        scalar_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1 = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            in_plane = fx.slice(in_buf, (token, None, None))
            out_plane = fx.slice(out_buf, (token, None, None))

            vals = [[None] * H for _ in range(H)]
            for j in range_constexpr(H):
                row_in = fx.slice(in_plane, (j, None))
                row_in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
                for k in range_constexpr(H):
                    r = fx.memref_alloca(scalar_ty, s_lay)
                    fx.copy_atom_call(copy_atom_32, fx.slice(row_in_div, (None, k)), r)
                    vals[j][k] = fx.Float32(
                        fxvec.extract(fx.memref_load_vec(r), static_position=[0])
                    )

            # softmax(dim=-1) per row; fold + eps
            for j in range_constexpr(H):
                rmax = vals[j][0]
                for k in range_constexpr(H - 1):
                    rmax = rmax.maximumf(vals[j][k + 1])
                for k in range_constexpr(H):
                    vals[j][k] = fx.Float32(fxmath.exp(vals[j][k] - rmax))
                rsum = vals[j][0]
                for k in range_constexpr(H - 1):
                    rsum = rsum + vals[j][k + 1]
                for k in range_constexpr(H):
                    vals[j][k] = vals[j][k] / rsum + fx.Float32(eps)

            # column normalize
            for k in range_constexpr(H):
                csum = vals[0][k]
                for j in range_constexpr(H - 1):
                    csum = csum + vals[j + 1][k]
                csum_eps = csum + fx.Float32(eps)
                for j in range_constexpr(H):
                    vals[j][k] = vals[j][k] / csum_eps

            # repeat - 1 iterations of (row_norm, col_norm)
            for _step in range_constexpr(repeat - 1):
                for j in range_constexpr(H):
                    rsum = vals[j][0]
                    for k in range_constexpr(H - 1):
                        rsum = rsum + vals[j][k + 1]
                    rsum_eps = rsum + fx.Float32(eps)
                    for k in range_constexpr(H):
                        vals[j][k] = vals[j][k] / rsum_eps
                for k in range_constexpr(H):
                    csum = vals[0][k]
                    for j in range_constexpr(H - 1):
                        csum = csum + vals[j + 1][k]
                    csum_eps = csum + fx.Float32(eps)
                    for j in range_constexpr(H):
                        vals[j][k] = vals[j][k] / csum_eps

            # store back
            for j in range_constexpr(H):
                row_out = fx.slice(out_plane, (j, None))
                row_out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
                for k in range_constexpr(H):
                    r = fx.memref_alloca(scalar_ty, s_lay)
                    fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [vals[j][k]]), r)
                    fx.copy_atom_call(copy_atom_32, r, fx.slice(row_out_div, (None, k)))

    @flyc.jit
    def launch(
        x_in:       fx.Tensor,
        x_out:      fx.Tensor,
        num_tokens: fx.Int32,
        stream:     fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        kernel(x_in, x_out, num_tokens).launch(
            grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
        )

    def runner(x_in, x_out, num_tokens):
        if num_tokens == 0:
            return
        launch(x_in.detach(), x_out.detach(), num_tokens)

    return runner


_KERNEL_CACHE: dict[tuple[int, int, float], object] = {}


def get_sinkhorn_kernel(hidden_size: int, repeat: int, eps: float):
    key = (int(hidden_size), int(repeat), float(eps))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_sinkhorn_kernel(*key)
    return _KERNEL_CACHE[key]
