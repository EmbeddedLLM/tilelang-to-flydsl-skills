"""FlyvLLM port of ``vllm.v1.attention.ops.rocm_aiter_mla_sparse``.

Public functions mirror the upstream signatures so call sites can be a
drop-in swap.  Internally the heavy compute is delegated to FlyDSL kernels
under ``fly_vllm.kernels.*``; thin orchestration code stays torch.

Target: gfx950 (CDNA 4 / MI350).  FP8 = OCP E4M3FN (not FNUZ).

Each compute primitive has:
- a torch reference (``*_torch``) lifted from upstream for testing,
- a FlyDSL fast path (``*_flydsl``) with the same signature,
- a public entry point that picks the FlyDSL path when CUDA is available.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FP8 dtype on gfx950 — always E4M3FN (OCP), never FNUZ.
# ---------------------------------------------------------------------------

FP8_DTYPE = torch.float8_e4m3fn


def _is_fnuz() -> bool:
    """gfx950 uses OCP fp8, not FNUZ."""
    return False


# ---------------------------------------------------------------------------
# E8M0 helpers (block-scale format used by the dequant path).
# ---------------------------------------------------------------------------

def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 (8-bit exponent only) → float32 by placing bits in exp field."""
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    fp32_bits = exp_bits << 23
    return fp32_bits.view(torch.float32)


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


# ---------------------------------------------------------------------------
# Forward declarations populated from the kernel modules below.
# ---------------------------------------------------------------------------

from fly_vllm.kernels.dequantize_blocked_k_cache import (
    rocm_dequantize_blocked_k_cache_flydsl,
    rocm_dequantize_blocked_k_cache_torch,
)
from fly_vllm.kernels.inv_rope import (
    apply_gptj_inv_rope_flydsl,
    apply_gptj_inv_rope_torch,
    rocm_inv_rope_einsum_flydsl,
    rocm_inv_rope_einsum_torch,
)
from fly_vllm.kernels.indexer_k_quant_and_cache import (
    indexer_k_quant_and_cache_flydsl,
    indexer_k_quant_and_cache_torch,
)
from fly_vllm.kernels.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache_flydsl,
    cp_gather_indexer_k_quant_cache_torch,
)
from fly_vllm.kernels.fp8_mqa_logits import (
    fp8_mqa_logits_flydsl,
    fp8_mqa_logits_torch,
)
from fly_vllm.kernels.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits_flydsl,
    fp8_paged_mqa_logits_torch,
)
from fly_vllm.kernels.topk_indices import (
    topk_indices_flydsl,
    topk_indices_torch,
)
from fly_vllm.kernels.sparse_attn_prefill import (
    rocm_ref_sparse_attn_prefill_flydsl,
    rocm_ref_sparse_attn_prefill_torch,
)
from fly_vllm.kernels.sparse_attn_decode import (
    rocm_ref_sparse_attn_decode_flydsl,
    rocm_ref_sparse_attn_decode_torch,
)


# ---------------------------------------------------------------------------
# Public entry points (signatures match upstream).
# ---------------------------------------------------------------------------

def rocm_dequantize_blocked_k_cache(
    quant_k_cache: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    return rocm_dequantize_blocked_k_cache_flydsl(
        quant_k_cache, head_dim, nope_head_dim, rope_head_dim
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    indexer_k_quant_and_cache_flydsl(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
        block_tile_size=block_tile_size,
        head_tile_size=head_tile_size,
    )


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
) -> None:
    cp_gather_indexer_k_quant_cache_flydsl(
        k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq,
        block_tile_size=block_tile_size, head_tile_size=head_tile_size,
    )


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    return fp8_mqa_logits_flydsl(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    max_model_len: int,
) -> torch.Tensor:
    return fp8_paged_mqa_logits_flydsl(
        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
    )


def topk_indices(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    return topk_indices_flydsl(logits, topk_tokens)


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    return rocm_inv_rope_einsum_flydsl(
        rotary_emb, o, positions, rope_head_dim, n_local_groups, o_lora_rank, wo_a
    )


def rocm_ref_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    return rocm_ref_sparse_attn_prefill_flydsl(
        q, kv, indices, topk_length, scale, head_dim, attn_sink
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    out = rocm_ref_sparse_attn_prefill_flydsl(
        q, kv, indices, topk_length, scale, head_dim, attn_sink
    )
    output.copy_(out.to(output.dtype))


def rocm_ref_sparse_attn_decode(
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    extra_blocked_k: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    return rocm_ref_sparse_attn_decode_flydsl(
        q, blocked_k, indices_in_kvcache, topk_length, scale, head_dim, attn_sink,
        extra_blocked_k=extra_blocked_k,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
    )


def rocm_forward_decode_fallback(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    blocked_swa = rocm_dequantize_blocked_k_cache_flydsl(
        swa_k_cache, head_dim=head_dim,
        nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
    )
    blocked_extra = None
    if not swa_only:
        assert kv_cache is not None
        blocked_extra = rocm_dequantize_blocked_k_cache_flydsl(
            kv_cache, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
        )
    attn_out = rocm_ref_sparse_attn_decode_flydsl(
        q=q.unsqueeze(1),
        blocked_k=blocked_swa,
        indices_in_kvcache=swa_indices.unsqueeze(1),
        topk_length=swa_lens,
        scale=scale,
        head_dim=head_dim,
        attn_sink=attn_sink[: q.shape[1]] if attn_sink is not None else None,
        extra_blocked_k=blocked_extra,
        extra_indices_in_kvcache=topk_indices,
        extra_topk_length=topk_lens,
    )
    output.copy_(attn_out.to(output.dtype))
