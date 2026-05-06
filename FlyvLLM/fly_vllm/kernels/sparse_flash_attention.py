"""sparse_flash_attention — port of tilelang-ascend's NPU example.

Source: ``tilelang-ascend/examples/deepseek_v4/sparse_flash_attention.py``.
The Ascend kernel provides FlashAttention with **sparse-gathered** KV
indexed by ``topk_idxs`` and a per-head ``attn_sink`` folded into the
softmax denominator.  The shapes are batched:

    q          : (b, m, h, d)        bf16
    kv         : (b, n, d)           bf16
    attn_sink  : (h,)                fp32
    topk_idxs  : (b, m, topk)        int32   (-1 means "invalid / pad")
    output     : (b, m, h, d)        fp32

Algorithm (matches ``sparse_attn`` in the upstream file bit-for-bit):

    sparse_kv = gather(kv, topk_idxs)   ; -1 indices materialise as 0
    score_mask = -inf where topk_idxs == -1 else 0
    scores = q @ sparse_kv.T            ; cast to fp32
    probs  = softmax(scores * scale + score_mask, dim=topk)
                with attn_sink folded into the denominator
    out    = probs @ sparse_kv          ; output stays fp32

The FlyDSL fast path reuses the FA2-style kernel from
``sparse_attn_prefill.py`` for size configs where its inner-loop unrolling
is tractable (d ≤ 256).  Larger configs (e.g. the upstream test's
``d=512``) currently fall back to the torch reference because the
fully-unrolled per-thread d-loop blows up compile time + register
pressure on gfx950 — the proper fix is an MFMA-based kernel (see the
project memory ``rocm_aiter_mla_sparse_port.md``).
"""

from __future__ import annotations

import torch

from fly_vllm.kernels.sparse_attn_prefill import (
    rocm_ref_sparse_attn_prefill_flydsl,
    rocm_ref_sparse_attn_prefill_torch,
    _can_use_flydsl as _prefill_can_use_flydsl,
    NUM_THREADS as _PREFILL_NUM_THREADS,
    BLOCK_N as _PREFILL_BLOCK_N,
)


# ---------------------------------------------------------------------------
# Torch reference (bit-for-bit copy of the helpers in the upstream file).
# ---------------------------------------------------------------------------

def gather_sparse_kv(kv_states: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    """Gather KV at ``topk_idxs`` per (batch, query); -1 indices map to zeros."""
    batch_size, seq_len, topk = topk_idxs.shape
    batch_idx = (
        torch.arange(batch_size, device=kv_states.device)
        .view(batch_size, 1, 1)
        .expand(-1, seq_len, topk)
    )
    safe = torch.where(topk_idxs == -1, torch.zeros_like(topk_idxs), topk_idxs).long()
    gathered = kv_states[batch_idx, safe, :]
    mask = (topk_idxs != -1).unsqueeze(-1).to(gathered.dtype)
    return gathered * mask


def sparse_softmax_with_sink(
    scores: torch.Tensor,
    attn_sink: torch.Tensor,
    head_dim: int,
    softmax_dim: int = -1,
) -> torch.Tensor:
    """Numerically-stable softmax with an extra ``attn_sink`` term in the
    denominator only.  ``head_dim`` is the *axis* (negative-friendly) along
    which ``attn_sink`` aligns to ``scores``."""
    max_vals = torch.max(scores, dim=softmax_dim, keepdim=True).values
    exp_scores = torch.exp(scores - max_vals)
    sum_exp = torch.sum(exp_scores, dim=softmax_dim, keepdim=True)
    sink_view_shape = [1] * scores.dim()
    sink_view_shape[head_dim if head_dim >= 0 else scores.dim() + head_dim] = scores.shape[head_dim]
    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    return exp_scores / (sum_exp + sink_term)


def sparse_attn_torch(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference path — computes everything in fp32 (output stays fp32)."""
    sparse_kv = gather_sparse_kv(kv_states, topk_idxs)
    score_mask = torch.where(
        (topk_idxs == -1).unsqueeze(-2),
        torch.tensor(-float("inf"), device=query_states.device, dtype=torch.float32),
        torch.tensor(0.0, device=query_states.device, dtype=torch.float32),
    )
    scores = torch.matmul(query_states, sparse_kv.transpose(-2, -1)).to(torch.float32)
    probs = sparse_softmax_with_sink(
        scores * softmax_scale + score_mask, attn_sink, head_dim=-2,
    )
    return torch.matmul(probs, sparse_kv.to(torch.float32))


# ---------------------------------------------------------------------------
# FlyDSL fast path.
# ---------------------------------------------------------------------------

def _can_use_flydsl(q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor) -> bool:
    """Eligibility for the per-(b, m) FlashAttention-2 FlyDSL kernel.

    Reuses the gate from ``sparse_attn_prefill`` after folding the batch
    dim into ``s_q``.  Excludes:
    - non-bf16 q/kv,
    - very large d (>256) where the unrolled inner loops blow up compile time,
    - topk shapes that aren't divisible by the kernel's BLOCK_N.
    """
    b, m, h, d = q.shape
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False
    if d > 256 or h * (d // _PREFILL_NUM_THREADS) <= 0:
        # head_dim must be a multiple of the kernel's NUM_THREADS, and the
        # per-thread output count must be ≥ 1.
        if d % _PREFILL_NUM_THREADS != 0:
            return False
    topk = topk_idxs.shape[-1]
    if topk % _PREFILL_BLOCK_N != 0:
        return False
    return d <= 256 and (d % _PREFILL_NUM_THREADS) == 0


def sparse_attn_flydsl(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """FlyDSL fast path with torch fallback for unsupported shapes."""
    b, m, h, d = query_states.shape
    if not _can_use_flydsl(query_states, kv_states, topk_idxs):
        return sparse_attn_torch(
            query_states, kv_states, attn_sink, topk_idxs, softmax_scale,
        )

    # Reshape (b, m) → s_q.  Globalize indices so each query in batch b
    # sees its own kv slice when we hand a flattened (b*n, d) kv tensor
    # to the per-(s_q, h_q) prefill kernel.
    n = kv_states.shape[1]
    q_flat = query_states.reshape(b * m, h, d).contiguous()
    kv_flat = kv_states.reshape(b * n, d).contiguous()
    # Build batch-offset for indices: each (b_i, m_i) batch's indices need
    # +b_i*n added.  -1 stays -1 (mask).
    bo = (torch.arange(b, dtype=torch.int32, device=topk_idxs.device) * n).view(b, 1, 1)
    indices_glob = torch.where(
        topk_idxs == -1, topk_idxs, topk_idxs + bo,
    )
    indices_glob = indices_glob.reshape(b * m, 1, -1).contiguous()

    out_bf = rocm_ref_sparse_attn_prefill_flydsl(
        q_flat, kv_flat, indices_glob, None,
        scale=softmax_scale,
        head_dim=d,
        attn_sink=attn_sink.to(torch.float32),
    )
    # rocm_ref_sparse_attn_prefill_flydsl returns bf16; cast to fp32 so we
    # match the upstream (and Ascend) output dtype.
    return out_bf.reshape(b, m, h, d).to(torch.float32)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def sparse_attn(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """FlashAttention with sparse-gathered KV + attn_sink (DSv4 layer).

    Drop-in for the upstream Ascend ``sparse_attn`` — same signature, same
    output dtype (fp32), same -1-index semantics.
    """
    return sparse_attn_flydsl(
        query_states, kv_states, attn_sink, topk_idxs, softmax_scale,
    )
