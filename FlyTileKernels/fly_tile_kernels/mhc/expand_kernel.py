"""mhc.expand: replicate the last-dim hidden vector `mhc_mult` times.

Forward: for each token (i), copy ``x[i, :]`` (bf16) to ``o[i, m, :]`` for
m in ``range(mhc_mult)``.  No math.

Backward: ``x_grad[i, j] = sum_m o_grad[i, m, j]`` accumulated in fp32 then
written back as bf16.

Layout choice:
- One token row per block on grid.x; one tile of ``BLK_H`` columns on
  grid.y; block has ``BLK_H`` threads (one column per thread).  This avoids
  the 32x128 register fragment in the TileLang version that would exceed
  the AMDGPU per-block thread cap.

API notes (this FlyDSL build, gfx950): see
``moe/normalize_weight_kernel.py`` for the underlying patterns
(``_dyn(...)`` for dynamic-if; per-element BufferCopy load/store; bf16
element width selection).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


BLK_H = 128


def _dyn(x):
    return x


def expand_to_mhc_fwd_tl(hidden: int, mhc_mult: int):
    h = hidden
    mhc = mhc_mult

    @flyc.kernel
    def fwd_kernel(
        x:           fx.Tensor,   # (n, h) bf16
        o:           fx.Tensor,   # (n, mhc, h) bf16
        num_tokens:  fx.Int32,
    ):
        i = fx.block_idx.x  # token row
        bid_y = fx.block_idx.y
        tid = fx.thread_idx.x
        j = bid_y * BLK_H + tid  # column

        x_buf = fx.rocdl.make_buffer_tensor(x)
        o_buf = fx.rocdl.make_buffer_tensor(o)

        # Per-element register slot.
        scalar_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        if _dyn(j < h):
            # x[i, j] → register
            row_view = fx.slice(x_buf, (i, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            elem_reg = fx.memref_alloca(scalar_ty, s_lay)
            fx.copy_atom_call(copy_atom_16, fx.slice(row_div, (None, j)), elem_reg)

            # Write mhc copies of x[i, j] into o[i, m, j].
            for m in range_constexpr(mhc):
                slab = fx.slice(o_buf, (i, m, None))  # 1-D length h
                slab_div = fx.logical_divide(slab, fx.make_layout(1, 1))
                fx.copy_atom_call(copy_atom_16, elem_reg, fx.slice(slab_div, (None, j)))

    @flyc.jit
    def launch(
        x:          fx.Tensor,
        o:          fx.Tensor,
        num_tokens: fx.Int32,
        stream:     fx.Stream = fx.Stream(None),
    ):
        gx = num_tokens
        gy = (h + BLK_H - 1) // BLK_H
        fwd_kernel(x, o, num_tokens).launch(
            grid=(gx, gy, 1), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(x_t: torch.Tensor, o_t: torch.Tensor):
        n = x_t.shape[0]
        if n == 0:
            return
        launch(x_t.detach(), o_t.detach(), n)

    return runner


def expand_to_mhc_bwd_tl(hidden: int, mhc_mult: int):
    h = hidden
    mhc = mhc_mult

    @flyc.kernel
    def bwd_kernel(
        o_grad:     fx.Tensor,   # (n, mhc, h) bf16
        x_grad:     fx.Tensor,   # (n, h) bf16
        num_tokens: fx.Int32,
    ):
        i = fx.block_idx.x
        bid_y = fx.block_idx.y
        tid = fx.thread_idx.x
        j = bid_y * BLK_H + tid

        og_buf = fx.rocdl.make_buffer_tensor(o_grad)
        xg_buf = fx.rocdl.make_buffer_tensor(x_grad)

        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        vec_ty_1_bf16 = ir.VectorType.get([1], fx.BFloat16.ir_type)

        if _dyn(j < h):
            acc = fx.Float32(0.0)
            for m in range_constexpr(mhc):
                slab = fx.slice(og_buf, (i, m, None))
                slab_div = fx.logical_divide(slab, fx.make_layout(1, 1))
                tmp = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(copy_atom_16, fx.slice(slab_div, (None, j)), tmp)
                e_bf = fxvec.extract(fx.memref_load_vec(tmp), static_position=[0])
                acc = acc + fx.BFloat16(e_bf).to(fx.Float32)

            # Cast back to bf16 and store.
            out_bf = acc.to(fx.BFloat16)
            out_reg = fx.memref_alloca(bf16_reg_ty, s_lay)
            out_vec = fxvec.from_elements(vec_ty_1_bf16, [out_bf])
            fx.memref_store_vec(out_vec, out_reg)
            x_row = fx.slice(xg_buf, (i, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
            fx.copy_atom_call(copy_atom_16, out_reg, fx.slice(x_row_div, (None, j)))

    @flyc.jit
    def launch(
        o_grad:     fx.Tensor,
        x_grad:     fx.Tensor,
        num_tokens: fx.Int32,
        stream:     fx.Stream = fx.Stream(None),
    ):
        gx = num_tokens
        gy = (h + BLK_H - 1) // BLK_H
        bwd_kernel(o_grad, x_grad, num_tokens).launch(
            grid=(gx, gy, 1), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(og_t: torch.Tensor, xg_t: torch.Tensor):
        n = og_t.shape[0]
        if n == 0:
            return
        launch(og_t.detach(), xg_t.detach(), n)

    return runner
