"""cp_gather_indexer_k_quant_cache: gather a paged FP8 KV cache by token.

Per token (one wave-lane each):
    batch_id  = token_to_seq[t]
    bs, be    = cu_seqlen[batch_id], cu_seqlen[batch_id+1]
    if t >= be: skip
    bo        = t - bs                       # batch-local offset
    btid      = bo // BS                     # block-table index
    bo_blk    = bo  % BS                     # in-block offset
    block_id  = block_table[batch_id, btid]
    src_addr  = SHUFFLE address for (block_id, bo_blk, h) inside cache_value
    k_fp8[t, h]   = cache_value[src_addr]
    k_scale[t, 0:4] = cache_scale[block_id, bo_blk]    (fp32 → 4 bytes)

One thread per token, head_dim copies sequentially in the inner loop.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def _build_gather_kernel(head_dim: int, block_size: int,
                         block_tile_size: int, head_tile_size: int):
    HD = head_dim
    BS = block_size
    BT = block_tile_size
    HT = head_tile_size

    @flyc.kernel
    def kernel(
        cache_value: fx.Tensor,    # (num_blocks, block_size * head_dim) uint8
        cache_scale: fx.Tensor,    # (num_blocks, block_size) fp32
        k_fp8_u8:    fx.Tensor,    # (num_tokens, head_dim) uint8 view of fp8
        k_scale_f32: fx.Tensor,    # (num_tokens,) fp32 view of the 4-byte scale
        block_table: fx.Tensor,    # (batch_size, ?) int32
        cu_seqlen:   fx.Tensor,    # (batch_size + 1,) int32
        token_to_seq: fx.Tensor,   # (num_tokens,) int32
        block_table_stride: fx.Int32,
        num_tokens:  fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        v_buf = fx.rocdl.make_buffer_tensor(cache_value)
        s_buf = fx.rocdl.make_buffer_tensor(cache_scale)
        kf_buf = fx.rocdl.make_buffer_tensor(k_fp8_u8)
        ks_buf = fx.rocdl.make_buffer_tensor(k_scale_f32)
        bt_buf = fx.rocdl.make_buffer_tensor(block_table)
        cu_buf = fx.rocdl.make_buffer_tensor(cu_seqlen)
        ts_buf = fx.rocdl.make_buffer_tensor(token_to_seq)

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
        atom_8_u8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)
        atom_32_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        atom_32_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1_u8 = ir.VectorType.get([1], fx.Uint8.ir_type)
        vec_ty_1_f = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            # batch_id = token_to_seq[token]
            ts_div = fx.logical_divide(ts_buf, fx.make_layout(1, 1))
            r = fx.memref_alloca(i32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_i, fx.slice(ts_div, (None, token)), r)
            batch_id = fx.Int32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))

            cu_div = fx.logical_divide(cu_buf, fx.make_layout(1, 1))
            r = fx.memref_alloca(i32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_i, fx.slice(cu_div, (None, batch_id)), r)
            bs = fx.Int32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
            r = fx.memref_alloca(i32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_i, fx.slice(cu_div, (None, batch_id + fx.Int32(1))), r)
            be = fx.Int32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))

            if _dyn(token < be):
                bo = token - bs
                btid = bo // fx.Int32(BS)
                boff = bo % fx.Int32(BS)
                tbid = boff // fx.Int32(BT)
                tboff = boff % fx.Int32(BT)
                base = tbid * fx.Int32(BT * HD) + tboff * fx.Int32(HT)

                # block_id = block_table[batch_id, btid]
                bt_row = fx.slice(bt_buf, (batch_id, None))
                bt_div = fx.logical_divide(bt_row, fx.make_layout(1, 1))
                r = fx.memref_alloca(i32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_i, fx.slice(bt_div, (None, btid)), r)
                block_id = fx.Int32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))

                # Copy fp8 bytes from cache → k_fp8.
                v_row = fx.slice(v_buf, (block_id, None))
                v_row_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
                kf_row = fx.slice(kf_buf, (token, None))
                kf_row_div = fx.logical_divide(kf_row, fx.make_layout(1, 1))
                for h in range_constexpr(HD):
                    tile_h = (h // HT) * (BT * HT) + (h % HT)
                    src_off = base + fx.Int32(tile_h)
                    rr = fx.memref_alloca(u8_reg_ty, s_lay)
                    fx.copy_atom_call(atom_8_u8, fx.slice(v_row_div, (None, src_off)), rr)
                    fx.copy_atom_call(atom_8_u8, rr, fx.slice(kf_row_div, (None, h)))

                # Copy scale.
                s_row = fx.slice(s_buf, (block_id, None))
                s_row_div = fx.logical_divide(s_row, fx.make_layout(1, 1))
                rrf = fx.memref_alloca(f32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_f, fx.slice(s_row_div, (None, boff)), rrf)
                ks_div = fx.logical_divide(ks_buf, fx.make_layout(1, 1))
                fx.copy_atom_call(atom_32_f, rrf, fx.slice(ks_div, (None, token)))

    @flyc.jit
    def launch(
        cache_value, cache_scale, k_fp8_u8, k_scale_f32,
        block_table, cu_seqlen, token_to_seq,
        block_table_stride, num_tokens,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        kernel(
            cache_value, cache_scale, k_fp8_u8, k_scale_f32,
            block_table, cu_seqlen, token_to_seq,
            block_table_stride, num_tokens,
        ).launch(grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    def runner(cache_value, cache_scale, k_fp8_u8, k_scale_f32,
               block_table, cu_seqlen, token_to_seq,
               block_table_stride, num_tokens):
        if num_tokens == 0:
            return
        launch(
            cache_value.detach(), cache_scale.detach(),
            k_fp8_u8.detach(), k_scale_f32.detach(),
            block_table.detach(), cu_seqlen.detach(),
            token_to_seq.detach(),
            int(block_table_stride), num_tokens,
        )

    return runner


_KERNEL_CACHE: dict[tuple, object] = {}


def _get_kernel(head_dim: int, block_size: int, block_tile: int, head_tile: int):
    key = (int(head_dim), int(block_size), int(block_tile), int(head_tile))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_gather_kernel(*key)
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Reference torch path (slow Python loop).
# ---------------------------------------------------------------------------

def cp_gather_indexer_k_quant_cache_torch(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    fp8_dtype = torch.float8_e4m3fn
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    head_dim = k_fp8.shape[-1]
    flat = k_cache.view(num_blocks, -1)
    cache_value = flat[:, : block_size * head_dim].view(fp8_dtype)
    cache_scale = flat[:, block_size * head_dim :].view(torch.float32)
    head_tile_e = head_tile_size // k_cache.element_size()
    num_tokens = k_fp8.size(0)
    k_fp8_scale_f = k_fp8_scale.view(torch.float32)

    for tid in range(num_tokens):
        batch_id = int(token_to_seq[tid].item())
        batch_start = int(cu_seqlen[batch_id].item())
        batch_end = int(cu_seqlen[batch_id + 1].item())
        if tid >= batch_end:
            continue
        bo = tid - batch_start
        btid = bo // block_size
        boff = bo % block_size
        block_id = int(block_table[batch_id, btid].item())
        tbid = boff // block_tile_size
        tboff = boff % block_tile_size
        idx = torch.arange(head_dim, device=k_fp8.device)
        tile_offset = (idx // head_tile_e) * (head_tile_e * block_tile_size) + (idx % head_tile_e)
        base = tbid * (head_dim * block_tile_size) + tboff * head_tile_e
        row = cache_value.view(num_blocks, -1)
        k_fp8[tid] = row[block_id, base + tile_offset].view(torch.float8_e4m3fn)
        k_fp8_scale_f[tid] = cache_scale[block_id, boff]


# ---------------------------------------------------------------------------
# FlyDSL public path.
# ---------------------------------------------------------------------------

def cp_gather_indexer_k_quant_cache_flydsl(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    head_dim = k_fp8.shape[-1]
    flat = k_cache.view(num_blocks, -1)
    cache_value_u8 = flat[:, : block_size * head_dim]
    cache_scale_f = flat[:, block_size * head_dim :].view(torch.float32)
    head_tile_e = head_tile_size // k_cache.element_size()
    num_tokens = k_fp8.size(0)
    if num_tokens == 0:
        return
    k_fp8_u8 = k_fp8.view(torch.uint8)
    k_fp8_scale_f = k_fp8_scale.view(torch.float32)
    runner = _get_kernel(head_dim, block_size, block_tile_size, head_tile_e)
    runner(
        cache_value_u8, cache_scale_f, k_fp8_u8, k_fp8_scale_f,
        block_table, cu_seqlen, token_to_seq,
        block_table.stride(0), num_tokens,
    )
