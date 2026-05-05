"""inverse RoPE + WO_A einsum.

The inverse-RoPE kernel mirrors GPT-J style rotary embeddings, applied with
the negated sin sign (inverse).  Per (token, group, even pair):
    even_in = x[t, g, nope+2*p]
    odd_in  = x[t, g, nope+2*p+1]
    even_out = even_in * cos[t, p] + odd_in  * sin[t, p]
    odd_out  = odd_in  * cos[t, p] - even_in * sin[t, p]
The nope prefix is copied through unchanged.

The kernel writes back to a fresh tensor in the input dtype.

The wrapper accepts both 2-D ``(t, head_dim)`` and 3-D ``(t, groups, head_dim)``
inputs, matching the caller in vllm.
"""

from __future__ import annotations

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


BLK_H = 256


def _dyn(x):
    return x


def _build_inv_rope_kernel(groups: int, head_dim: int, rope_dim: int):
    G = groups
    HD = head_dim
    RD = rope_dim
    NHD = HD - RD
    HALF = RD // 2

    @flyc.kernel
    def kernel(
        x_bf:    fx.Tensor,    # (n, G, HD) bf16
        cos_sin: fx.Tensor,    # (max_pos, RD) fp32  (cos[:HALF] | sin[HALF:RD])
        positions: fx.Tensor,  # (n,) int32
        out_bf:  fx.Tensor,    # (n, G, HD) bf16
        num_tokens: fx.Int32,
    ):
        token = fx.block_idx.x
        bid_h = fx.block_idx.y
        group = fx.block_idx.z
        tid = fx.thread_idx.x
        h_idx = bid_h * BLK_H + tid

        x_buf = fx.rocdl.make_buffer_tensor(x_bf)
        cs_buf = fx.rocdl.make_buffer_tensor(cos_sin)
        p_buf = fx.rocdl.make_buffer_tensor(positions)
        o_buf = fx.rocdl.make_buffer_tensor(out_bf)

        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        f32_reg_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        i32_reg_ty = fx.MemRefType.get(
            fx.Int32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        atom_32_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        atom_32_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        vec_ty_1_bf = ir.VectorType.get([1], fx.BFloat16.ir_type)

        if _dyn((token < num_tokens) & (h_idx < HD)):
            o_row = fx.slice(o_buf, (token, group, None))
            o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
            x_row = fx.slice(x_buf, (token, group, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))

            if _dyn(h_idx < NHD):
                # nope passthrough
                rr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(x_row_div, (None, h_idx)), rr)
                fx.copy_atom_call(atom_16, rr, fx.slice(o_row_div, (None, h_idx)))
            else:
                rh = h_idx - NHD
                pair = rh // fx.Int32(2)
                is_odd = (rh % fx.Int32(2)) != fx.Int32(0)

                # Position for this token
                p_div = fx.logical_divide(p_buf, fx.make_layout(1, 1))
                pr = fx.memref_alloca(i32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_i, fx.slice(p_div, (None, token)), pr)
                pos = fx.Int32(fxvec.extract(fx.memref_load_vec(pr), static_position=[0]))

                cs_row = fx.slice(cs_buf, (pos, None))
                cs_div = fx.logical_divide(cs_row, fx.make_layout(1, 1))
                # cos[pair]
                cr = fx.memref_alloca(f32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_f, fx.slice(cs_div, (None, pair)), cr)
                cos = fx.Float32(fxvec.extract(fx.memref_load_vec(cr), static_position=[0]))
                # sin[pair]
                sr = fx.memref_alloca(f32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_f, fx.slice(cs_div, (None, pair + fx.Int32(HALF))), sr)
                sin = fx.Float32(fxvec.extract(fx.memref_load_vec(sr), static_position=[0]))

                # Load even and odd siblings (paired index).
                even_idx = NHD + pair * fx.Int32(2)
                odd_idx = even_idx + fx.Int32(1)
                er = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(x_row_div, (None, even_idx)), er)
                even_v = fx.BFloat16(
                    fxvec.extract(fx.memref_load_vec(er), static_position=[0])
                ).to(fx.Float32)
                odr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(x_row_div, (None, odd_idx)), odr)
                odd_v = fx.BFloat16(
                    fxvec.extract(fx.memref_load_vec(odr), static_position=[0])
                ).to(fx.Float32)

                even_out = even_v * cos + odd_v * sin
                odd_out = odd_v * cos - even_v * sin
                # select per-thread output
                out_v = is_odd.select(odd_out.ir_value(), even_out.ir_value())
                out_bf16 = fx.Float32(out_v).to(fx.BFloat16)
                outr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1_bf, [out_bf16]), outr)
                fx.copy_atom_call(atom_16, outr, fx.slice(o_row_div, (None, h_idx)))

    @flyc.jit
    def launch(
        x_bf, cos_sin, positions, out_bf, num_tokens,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = num_tokens
        gy = (HD + BLK_H - 1) // BLK_H
        gz = G
        kernel(x_bf, cos_sin, positions, out_bf, num_tokens).launch(
            grid=(gx, gy, gz), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(x_bf, cos_sin, positions, out_bf, num_tokens):
        if num_tokens == 0:
            return
        launch(
            x_bf.detach(), cos_sin.detach(),
            positions.detach(), out_bf.detach(), num_tokens,
        )

    return runner


_KERNEL_CACHE: dict[tuple, object] = {}


def _get_kernel(groups: int, head_dim: int, rope_dim: int):
    key = (int(groups), int(head_dim), int(rope_dim))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_inv_rope_kernel(*key)
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Reference implementations.
# ---------------------------------------------------------------------------

def apply_gptj_inv_rope_torch(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def apply_gptj_inv_rope_flydsl(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    assert x.dtype == torch.bfloat16, "inv_rope flydsl path expects bf16 input"
    head_dim = x.shape[-1]
    if x.dim() == 2:
        # Pretend a single group.
        x3 = x.unsqueeze(1)
        squeeze_back = True
    else:
        x3 = x
        squeeze_back = False
    n, g, _ = x3.shape
    out3 = torch.empty_like(x3)
    runner = _get_kernel(g, head_dim, rope_dim)
    if positions.dtype != torch.int32:
        positions = positions.to(torch.int32)
    runner(x3.contiguous(), cos_sin_cache.to(torch.float32).contiguous(),
           positions, out3, n)
    return out3.squeeze(1) if squeeze_back else out3


def _expand_2d_block_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        exp_bits = scale.view(torch.uint8).to(torch.int32)
        scale = (exp_bits << 23).view(torch.float32)
    else:
        scale = scale.to(torch.float32)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def rocm_inv_rope_einsum_torch(
    rotary_emb,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a,
) -> torch.Tensor:
    o_ref = apply_gptj_inv_rope_torch(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    ).to(torch.bfloat16)
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)
    hidden_dim = o_ref.shape[-1]
    if hasattr(wo_a, "weight_scale_inv"):
        w = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.float32)
        s = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank, hidden_dim,
        )
        w = (w * s).to(torch.bfloat16)
    else:
        w = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    return torch.einsum("tgd,grd->tgr", o_ref, w)


def rocm_inv_rope_einsum_flydsl(
    rotary_emb,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a,
) -> torch.Tensor:
    """FlyDSL inv-RoPE then torch.einsum (the einsum is highly tuned in
    hipBLASLt/torch and isn't the bottleneck — only the RoPE step gets a
    handcoded path)."""
    o_rot = apply_gptj_inv_rope_flydsl(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    )
    o_rot = o_rot.view(o.shape[0], n_local_groups, -1)
    hidden_dim = o_rot.shape[-1]
    if hasattr(wo_a, "weight_scale_inv"):
        w = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.float32)
        s = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank, hidden_dim,
        )
        w = (w * s).to(torch.bfloat16)
    else:
        w = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    return torch.einsum("tgd,grd->tgr", o_rot, w)
