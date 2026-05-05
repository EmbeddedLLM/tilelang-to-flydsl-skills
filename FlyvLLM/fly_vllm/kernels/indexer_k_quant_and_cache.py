"""indexer_k_quant_and_cache: per-token bf16-K → fp8-K quantize + scatter.

Per token (one wave-lane each):
    amax  = max_h |k[h]|
    scale = max(1e-4, amax) / 448.0                  (gfx950 OCP fp8 = 448)
    if UE8M0: scale = 2 ** ceil(log2(scale))
    fp8_v = (k[h] / scale) cast to fp8 e4m3fn
    write fp8_v into kv_cache at SHUFFLE address; write scale.

SHUFFLE address (matches the upstream Triton kernel):
    base        = (block_offset // BLOCK_TILE) * BLOCK_TILE * head_dim
                + (block_offset  % BLOCK_TILE) * HEAD_TILE
    tile_off[h] = (h // HEAD_TILE) * (BLOCK_TILE * HEAD_TILE) + (h % HEAD_TILE)
    cache_value[block_id, base + tile_off[h]] = fp8_v
    cache_scale[block_id, block_offset]       = scale

This kernel uses one thread per token (NUM_THREADS tokens per block).  Each
thread loops over h sequentially — head_dim is small (128) so the per-thread
work is bounded.
"""

from __future__ import annotations

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, math as fxmath
from flydsl._mlir import ir


NUM_THREADS = 128
FP8_MAX = 448.0  # OCP e4m3fn max representable
EPS = 1e-4


def _dyn(x):
    return x


def _build_indexer_kernel(
    head_dim: int, block_size: int,
    block_tile_size: int, head_tile_size: int,
    use_ue8m0: bool,
):
    HD = head_dim
    BS = block_size
    BT = block_tile_size
    HT = head_tile_size

    @flyc.kernel
    def kernel(
        k_bf16:       fx.Tensor,    # (num_tokens, head_dim) bf16
        cache_value:  fx.Tensor,    # (num_blocks, block_size * head_dim) uint8 (fp8 bits)
        cache_scale:  fx.Tensor,    # (num_blocks, block_size) fp32
        slot_mapping: fx.Tensor,    # (num_tokens,) int32
        num_tokens:   fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        k_buf = fx.rocdl.make_buffer_tensor(k_bf16)
        v_buf = fx.rocdl.make_buffer_tensor(cache_value)
        s_buf = fx.rocdl.make_buffer_tensor(cache_scale)
        slot_buf = fx.rocdl.make_buffer_tensor(slot_mapping)

        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        u8_reg_ty = fx.MemRefType.get(
            fx.Uint8.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        i32_reg_ty = fx.MemRefType.get(
            fx.Int32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        f32_reg_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        atom_8_u8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)
        atom_32_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        atom_32_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1_u8 = ir.VectorType.get([1], fx.Uint8.ir_type)
        vec_ty_1_f = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            # Load slot.
            slot_div = fx.logical_divide(slot_buf, fx.make_layout(1, 1))
            sr = fx.memref_alloca(i32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_i, fx.slice(slot_div, (None, token)), sr)
            slot = fx.Int32(fxvec.extract(fx.memref_load_vec(sr), static_position=[0]))

            if _dyn(slot >= fx.Int32(0)):
                block_id = slot // fx.Int32(BS)
                block_offset = slot % fx.Int32(BS)
                tbid = block_offset // fx.Int32(BT)
                tboff = block_offset % fx.Int32(BT)
                base = tbid * fx.Int32(BT * HD) + tboff * fx.Int32(HT)

                # Pass 1: load all bf16, compute amax in registers.
                k_row = fx.slice(k_buf, (token, None))
                k_row_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
                amax = fx.Float32(0.0)
                vals_f = []
                for h in range_constexpr(HD):
                    rr = fx.memref_alloca(bf16_reg_ty, s_lay)
                    fx.copy_atom_call(atom_16, fx.slice(k_row_div, (None, h)), rr)
                    bv = fx.BFloat16(
                        fxvec.extract(fx.memref_load_vec(rr), static_position=[0])
                    )
                    fv = bv.to(fx.Float32)
                    vals_f.append(fv)
                    av = fv.maximumf(fx.Float32(0.0) - fv)  # |x|
                    amax = amax.maximumf(av)

                # scale = max(EPS, amax) / 448.
                scale = amax.maximumf(fx.Float32(EPS)) / fx.Float32(FP8_MAX)
                # UE8M0 path is selected by the closure's `use_ue8m0` bool — no
                # FlyDSL Call so the AST rewriter does not promote it to a
                # dynamic if (we want compile-time branch elimination).
                if use_ue8m0:
                    log2s = fx.Float32(fxmath.log2(scale))
                    ceil_log2 = fx.Float32(fxmath.ceil(log2s))
                    scale = fx.Float32(fxmath.exp2(ceil_log2))

                # Pass 2: f32 → fp8 e4m3fn via the rocdl pack op (the generic
                # arith.truncf lowering for f8E4M3FN does not exist at LLVM IR
                # level on this build).  We pack one element at a time.
                # Match the torch reference's rounding by doing the *division*
                # rather than multiplying by an inv_scale (one fewer rounding).
                v_row = fx.slice(v_buf, (block_id, None))
                v_row_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
                old0 = fx.Uint32(0).ir_value()
                ff = fx.Uint32(0xFF).ir_value()
                for h in range_constexpr(HD):
                    scaled = vals_f[h] / scale
                    packed = fx.rocdl.cvt_pk_fp8_f32(
                        res=fx.Uint32.ir_type,
                        src_a=scaled.ir_value(),
                        src_b=scaled.ir_value(),
                        old=old0,
                        word_sel=False,
                    )
                    byte32 = packed & ff
                    byte_v = fx.Uint32(byte32).to(fx.Uint8)
                    tile_h = (h // HT) * (BT * HT) + (h % HT)
                    dst_off = base + fx.Int32(tile_h)
                    outr = fx.memref_alloca(u8_reg_ty, s_lay)
                    fx.memref_store_vec(fxvec.from_elements(vec_ty_1_u8, [byte_v]), outr)
                    fx.copy_atom_call(atom_8_u8, outr, fx.slice(v_row_div, (None, dst_off)))

                # Write scale.
                s_row = fx.slice(s_buf, (block_id, None))
                s_row_div = fx.logical_divide(s_row, fx.make_layout(1, 1))
                outr = fx.memref_alloca(f32_reg_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1_f, [scale]), outr)
                fx.copy_atom_call(atom_32_f, outr, fx.slice(s_row_div, (None, block_offset)))

    @flyc.jit
    def launch(
        k_bf16, cache_value, cache_scale, slot_mapping, num_tokens,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        kernel(k_bf16, cache_value, cache_scale, slot_mapping, num_tokens).launch(
            grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
        )

    def runner(k_bf16, cache_value, cache_scale, slot_mapping, num_tokens):
        if num_tokens == 0:
            return
        launch(
            k_bf16.detach(), cache_value.detach(), cache_scale.detach(),
            slot_mapping.detach(), num_tokens,
        )

    return runner


def const_expr_eq(a, b):
    """Tiny helper: like ``const_expr`` test but returns a python bool."""
    return a == b


_KERNEL_CACHE: dict[tuple, object] = {}


def _get_kernel(head_dim: int, block_size: int, block_tile: int, head_tile: int, ue8m0: bool):
    key = (int(head_dim), int(block_size), int(block_tile), int(head_tile), bool(ue8m0))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_indexer_kernel(*key)
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Reference torch path (slow Python loop).
# ---------------------------------------------------------------------------

def indexer_k_quant_and_cache_torch(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    fp8_dtype = torch.float8_e4m3fn
    use_ue8m0 = scale_fmt == "ue8m0"
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    flat = kv_cache.view(num_blocks, -1)
    cache_value = flat[:, : block_size * head_dim].view(fp8_dtype)
    cache_scale = flat[:, block_size * head_dim :].view(torch.float32)
    head_tile_e = head_tile_size // kv_cache.element_size()
    num_tokens = slot_mapping.shape[0]
    for tid in range(num_tokens):
        slot_id = int(slot_mapping[tid].item())
        if slot_id < 0:
            continue
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        tbid = block_offset // block_tile_size
        tboff = block_offset % block_tile_size
        val = k[tid].to(torch.float32)
        amax = float(val.abs().max().item())
        scale = max(EPS, amax) / FP8_MAX
        if use_ue8m0:
            scale = float(2 ** math.ceil(math.log2(scale)))
        fp8_val = (val / scale).to(fp8_dtype)
        idx = torch.arange(head_dim, device=k.device)
        tile_offset = (idx // head_tile_e) * (block_tile_size * head_tile_e) + (idx % head_tile_e)
        row = cache_value.view(num_blocks, -1)
        base = tbid * (block_tile_size * head_dim) + tboff * head_tile_e
        row[block_id, base + tile_offset] = fp8_val
        cache_scale[block_id, block_offset] = scale


# ---------------------------------------------------------------------------
# FlyDSL public path.
# ---------------------------------------------------------------------------

def indexer_k_quant_and_cache_flydsl(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    assert k.dtype == torch.bfloat16, "expected bf16 K input"
    fp8_dtype = torch.float8_e4m3fn
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    flat = kv_cache.view(num_blocks, -1)
    # cache_value as raw uint8 of shape (num_blocks, block_size*head_dim)
    cache_value_u8 = flat[:, : block_size * head_dim]
    cache_scale_f = flat[:, block_size * head_dim :].view(torch.float32)
    head_tile_e = head_tile_size // kv_cache.element_size()
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    use_ue8m0 = scale_fmt == "ue8m0"
    runner = _get_kernel(head_dim, block_size, block_tile_size, head_tile_e, use_ue8m0)
    runner(k.contiguous(), cache_value_u8, cache_scale_f, slot_mapping, num_tokens)
