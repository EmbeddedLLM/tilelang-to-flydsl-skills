"""pre_apply_mix: weighted-sum collapse over the mhc dimension.

Mirrors the ``_pre_apply_mix_fwd`` portion of
``mhc_pre_big_fuse_tilelang``:

    layer_input[t, h] = bfloat16( sum_m pre_mix[t, m] * float(residual[t, m, h]) )

One thread per (token, h) hidden-element pair, mhc unrolled in registers.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


BLK_H = 256


def _dyn(x):
    return x


def _build_pre_apply_mix_kernel(mhc_mult: int, hidden: int):
    h = hidden
    M = mhc_mult

    @flyc.kernel
    def kernel(
        residual:    fx.Tensor,   # (n, mhc, h) bf16
        pre_mix:     fx.Tensor,   # (n, mhc)    fp32
        layer_input: fx.Tensor,   # (n, h)      bf16
        num_tokens:  fx.Int32,
    ):
        token = fx.block_idx.x
        bid_h = fx.block_idx.y
        tid = fx.thread_idx.x
        h_idx = bid_h * BLK_H + tid

        x_buf = fx.rocdl.make_buffer_tensor(residual)
        m_buf = fx.rocdl.make_buffer_tensor(pre_mix)
        o_buf = fx.rocdl.make_buffer_tensor(layer_input)

        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        fp32_reg_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1_bf = ir.VectorType.get([1], fx.BFloat16.ir_type)

        if _dyn(h_idx < h):
            # Load per-token mix vector (length M) into M fp32 scalars.
            m_row = fx.slice(m_buf, (token, None))
            m_row_div = fx.logical_divide(m_row, fx.make_layout(1, 1))
            mix_vals = []
            for k in range_constexpr(M):
                r = fx.memref_alloca(fp32_reg_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(m_row_div, (None, k)), r)
                mix_vals.append(
                    fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                )

            # acc = sum_k mix[k] * float(x[token, k, h_idx])
            acc = fx.Float32(0.0)
            for k in range_constexpr(M):
                x_row = fx.slice(x_buf, (token, k, None))
                x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
                xr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(copy_atom_16, fx.slice(x_row_div, (None, h_idx)), xr)
                xv_bf = fxvec.extract(fx.memref_load_vec(xr), static_position=[0])
                xv_f = fx.BFloat16(xv_bf).to(fx.Float32)
                acc = acc + mix_vals[k] * xv_f

            out_bf = acc.to(fx.BFloat16)
            o_row = fx.slice(o_buf, (token, None))
            o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
            outr = fx.memref_alloca(bf16_reg_ty, s_lay)
            fx.memref_store_vec(fxvec.from_elements(vec_ty_1_bf, [out_bf]), outr)
            fx.copy_atom_call(copy_atom_16, outr, fx.slice(o_row_div, (None, h_idx)))

    @flyc.jit
    def launch(
        residual:    fx.Tensor,
        pre_mix:     fx.Tensor,
        layer_input: fx.Tensor,
        num_tokens:  fx.Int32,
        stream:      fx.Stream = fx.Stream(None),
    ):
        gx = num_tokens
        gy = (h + BLK_H - 1) // BLK_H
        kernel(residual, pre_mix, layer_input, num_tokens).launch(
            grid=(gx, gy, 1), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(residual, pre_mix, layer_input, num_tokens):
        if num_tokens == 0:
            return
        launch(residual.detach(), pre_mix.detach(), layer_input.detach(), num_tokens)

    return runner


_KERNEL_CACHE: dict[tuple[int, int], object] = {}


def get_pre_apply_mix_kernel(mhc_mult: int, hidden: int):
    key = (int(mhc_mult), int(hidden))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_pre_apply_mix_kernel(*key)
    return _KERNEL_CACHE[key]
