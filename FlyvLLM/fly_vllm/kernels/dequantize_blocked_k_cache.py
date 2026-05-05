"""rocm_dequantize_blocked_k_cache: FlyDSL port.

Layout of ``quant_k_cache``  shape (num_blocks, block_size, packed_uint8) where
packed_uint8 holds, per (block, pos):

    [ nope (fp8 e4m3fn, nope_head_dim bytes)
    | rope (bf16, 2 * rope_head_dim bytes)
    | scale (e8m0, 8 bytes ; only the first num_tiles are used)
    ]

Output shape: (num_blocks, block_size, 1, head_dim) bf16.

Math:
    out[..., :nope_head_dim] = bf16(fp32(nope) * fp32(scale_per_64-tile))
    out[..., nope_head_dim:] = bf16(rope)

The kernel is bandwidth-bound; one thread per (block, pos, h) output element.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


BLK_H = 256
TILE = 64


def _dyn(x):
    return x


def _build_dequant_kernel(block_size: int, nope_head_dim: int,
                          rope_head_dim: int, head_dim: int):
    BS = block_size
    NHD = nope_head_dim
    HD = head_dim

    @flyc.kernel
    def kernel(
        nope_u8:  fx.Tensor,    # (num_blocks, block_size, nope_head_dim) uint8 (fp8 e4m3fn bits)
        rope_bf:  fx.Tensor,    # (num_blocks, block_size, rope_head_dim) bf16
        scale_u8: fx.Tensor,    # (num_blocks, block_size, num_tiles)    uint8 (e8m0 bits)
        out_bf:   fx.Tensor,    # (num_blocks, block_size, head_dim)     bf16
        num_blocks: fx.Int32,
    ):
        bp = fx.block_idx.x
        bid_h = fx.block_idx.y
        tid = fx.thread_idx.x
        h_idx = bid_h * BLK_H + tid
        block = bp // BS
        pos = bp % BS

        n_buf = fx.rocdl.make_buffer_tensor(nope_u8)
        r_buf = fx.rocdl.make_buffer_tensor(rope_bf)
        s_buf = fx.rocdl.make_buffer_tensor(scale_u8)
        o_buf = fx.rocdl.make_buffer_tensor(out_bf)

        u8_reg_ty = fx.MemRefType.get(
            fx.Uint8.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        atom_8_u8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)
        atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        vec_ty_1_bf = ir.VectorType.get([1], fx.BFloat16.ir_type)

        if _dyn((block < num_blocks) & (h_idx < HD)):
            o_row = fx.slice(o_buf, (block, pos, None))
            o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
            outr = fx.memref_alloca(bf16_reg_ty, s_lay)

            if _dyn(h_idx < NHD):
                tile = h_idx // TILE
                # fp8 e4m3fn byte: decode to fp32 with bit ops.
                #   sign = (b>>7)&1 ; e=(b>>3)&0xF ; m=b&7
                #   normal:    val = (-1)^s * (8+m) * 2^(e-10)
                #   subnormal: val = (-1)^s * m * 2^-9         (e == 0)
                # Constructed fp32_bits directly:
                #   normal:    bits = (s<<31) | ((e+120)<<23) | (m<<20)
                #   subnormal m>0: pre-normalise m to put leading 1 at bit 3
                # Pure-arithmetic version (avoids the subnormal pre-normalise
                # entirely): val_f32 = sign * mant_eff * 2^(exp_eff)
                #   normal:    mant_eff = 8+m,  exp_eff = e-10
                #   subnormal: mant_eff = m,    exp_eff = -9
                n_row = fx.slice(n_buf, (block, pos, None))
                n_row_div = fx.logical_divide(n_row, fx.make_layout(1, 1))
                nr = fx.memref_alloca(u8_reg_ty, s_lay)
                fx.copy_atom_call(atom_8_u8, fx.slice(n_row_div, (None, h_idx)), nr)
                n_v = fxvec.extract(fx.memref_load_vec(nr), static_position=[0])
                n_b32 = fx.Uint8(n_v).to(fx.Uint32)
                bv = n_b32.ir_value()
                sign_bit = bv >> fx.Uint32(7).ir_value()
                sign_bit_l = sign_bit & fx.Uint32(1).ir_value()
                e_bits = (bv >> fx.Uint32(3).ir_value()) & fx.Uint32(0xF).ir_value()
                m_bits = bv & fx.Uint32(7).ir_value()
                e_i = fx.Uint32(e_bits).to(fx.Int32)
                m_i = fx.Uint32(m_bits).to(fx.Int32)
                is_norm = e_i > fx.Int32(0)
                mant_eff_i = fx.Int32(is_norm.select(m_i + fx.Int32(8), m_i))
                exp_eff_i = fx.Int32(is_norm.select(e_i - fx.Int32(10), fx.Int32(-9)))
                mant_eff = mant_eff_i.to(fx.Float32)
                exp_eff = exp_eff_i.to(fx.Float32)
                sign_f = fx.Float32(1.0) - fx.Uint32(sign_bit_l).to(fx.Float32) * fx.Float32(2.0)
                pow2 = exp_eff.exp2()
                n_f32 = sign_f * mant_eff * pow2
                # e8m0 scale byte: place 8-bit exponent into fp32 exponent field.
                s_row = fx.slice(s_buf, (block, pos, None))
                s_row_div = fx.logical_divide(s_row, fx.make_layout(1, 1))
                sr = fx.memref_alloca(u8_reg_ty, s_lay)
                fx.copy_atom_call(atom_8_u8, fx.slice(s_row_div, (None, tile)), sr)
                s_v = fxvec.extract(fx.memref_load_vec(sr), static_position=[0])
                s_b32 = fx.Uint8(s_v).to(fx.Uint32)
                s_bits = s_b32.ir_value() << fx.Uint32(23).ir_value()
                s_f32 = fx.Float32(s_bits.bitcast(fx.Float32.ir_type))
                val_bf = (n_f32 * s_f32).to(fx.BFloat16)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1_bf, [val_bf]), outr)
                fx.copy_atom_call(atom_16, outr, fx.slice(o_row_div, (None, h_idx)))
            else:
                r_h = h_idx - NHD
                r_row = fx.slice(r_buf, (block, pos, None))
                r_row_div = fx.logical_divide(r_row, fx.make_layout(1, 1))
                rr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(r_row_div, (None, r_h)), rr)
                fx.copy_atom_call(atom_16, rr, fx.slice(o_row_div, (None, h_idx)))

    @flyc.jit
    def launch(
        nope_u8:  fx.Tensor,
        rope_bf:  fx.Tensor,
        scale_u8: fx.Tensor,
        out_bf:   fx.Tensor,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = num_blocks * BS
        gy = (HD + BLK_H - 1) // BLK_H
        kernel(nope_u8, rope_bf, scale_u8, out_bf, num_blocks).launch(
            grid=(gx, gy, 1), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(nope_u8, rope_bf, scale_u8, out_bf, num_blocks):
        if num_blocks == 0:
            return
        launch(nope_u8.detach(), rope_bf.detach(), scale_u8.detach(),
               out_bf.detach(), num_blocks)

    return runner


_KERNEL_CACHE: dict[tuple[int, int, int, int], object] = {}


def _get_kernel(block_size: int, nope_head_dim: int, rope_head_dim: int, head_dim: int):
    key = (block_size, nope_head_dim, rope_head_dim, head_dim)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_dequant_kernel(*key)
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Reference (lifted verbatim from upstream).
# ---------------------------------------------------------------------------

def rocm_dequantize_blocked_k_cache_torch(
    quant_k_cache: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn
    tile_size = 64
    num_tiles = nope_head_dim // tile_size
    num_blocks, block_size, _ = quant_k_cache.shape
    flat = quant_k_cache.view(num_blocks, -1)
    nope_rope = flat[:, : block_size * (nope_head_dim + 2 * rope_head_dim)].view(
        num_blocks, block_size, nope_head_dim + 2 * rope_head_dim
    )
    nope = nope_rope[:, :, :nope_head_dim].view(fp8_dtype)
    rope = nope_rope[:, :, nope_head_dim:].view(torch.bfloat16)
    scale = (
        flat[:, block_size * (nope_head_dim + 2 * rope_head_dim) :]
        .view(num_blocks, block_size, 8)[:, :, :num_tiles]
        .view(torch.float8_e8m0fnu)
    )
    out = torch.empty(
        (num_blocks, block_size, 1, head_dim),
        dtype=torch.bfloat16, device=quant_k_cache.device,
    )
    out[..., nope_head_dim:] = rope.unsqueeze(2)
    for tile_idx in range(num_tiles):
        cur = nope[..., tile_idx * tile_size : (tile_idx + 1) * tile_size].to(torch.bfloat16)
        sc = scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        out[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (cur * sc).unsqueeze(2)
    return out


# ---------------------------------------------------------------------------
# FlyDSL public path.
# ---------------------------------------------------------------------------

def rocm_dequantize_blocked_k_cache_flydsl(
    quant_k_cache: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn
    num_tiles = nope_head_dim // TILE

    num_blocks, block_size, _ = quant_k_cache.shape
    flat = quant_k_cache.view(num_blocks, -1)
    nope_rope = flat[:, : block_size * (nope_head_dim + 2 * rope_head_dim)].view(
        num_blocks, block_size, nope_head_dim + 2 * rope_head_dim
    )
    # Pass FP8 nope as raw uint8 — the kernel bitcasts each byte to f8E4M3FN
    # internally, sidestepping the runtime DLPack adaptor's lack of f8E8M0FNU
    # support and the buffer-descriptor type constraint of copy_atom_call.
    nope_u8 = nope_rope[:, :, :nope_head_dim].contiguous()
    rope_bf = nope_rope[:, :, nope_head_dim:].contiguous().view(torch.bfloat16)
    scale_u8 = (
        flat[:, block_size * (nope_head_dim + 2 * rope_head_dim) :]
        .view(num_blocks, block_size, 8)[:, :, :num_tiles]
        .contiguous()
    )

    out = torch.empty(
        (num_blocks, block_size, head_dim),
        dtype=torch.bfloat16, device=quant_k_cache.device,
    )
    runner = _get_kernel(block_size, nope_head_dim, rope_head_dim, head_dim)
    runner(nope_u8, rope_bf, scale_u8, out, num_blocks)
    return out.view(num_blocks, block_size, 1, head_dim)
