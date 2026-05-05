"""sparse_attn_decode.

Decode-side sparse attention with optional second scope (e.g. SWA cache).
Like the prefill counterpart, GEMM-dominated.  Currently uses the torch
reference as the FlyDSL fast path.
"""

from __future__ import annotations

import torch


def rocm_ref_sparse_attn_decode_torch(
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
    b, s_q, h_q, d_qk = q.shape

    def process_scope(cur_blocked_k, cur_indices, cur_topk_length):
        cur_indices = cur_indices.reshape(b, s_q, -1)
        topk = cur_indices.size(-1)
        fixed = torch.clamp_min(cur_indices, 0)
        gathered = (
            cur_blocked_k.view(-1, d_qk)
            .index_select(0, fixed.view(-1))
            .view(b, s_q, topk, d_qk)
        )
        invalid = cur_indices == -1
        if cur_topk_length is not None:
            cur_topk_length = cur_topk_length.reshape(b)
            invalid |= torch.arange(0, topk, device=invalid.device).view(
                1, 1, topk
            ) >= cur_topk_length.view(b, 1, 1)
        return gathered, invalid

    gathered, invalid = process_scope(blocked_k, indices_in_kvcache, topk_length)
    if extra_blocked_k is not None:
        assert extra_indices_in_kvcache is not None
        g1, i1 = process_scope(extra_blocked_k, extra_indices_in_kvcache, extra_topk_length)
        gathered = torch.cat([gathered, g1], dim=2)
        invalid = torch.cat([invalid, i1], dim=2)
    gathered = gathered.view(b * s_q, -1, d_qk).float()
    gathered[gathered != gathered] = 0.0
    qf = q.float().view(b * s_q, h_q, d_qk)
    aw = qf @ gathered.transpose(-1, -2)
    aw *= scale
    aw[invalid.view(b * s_q, 1, -1).expand(b * s_q, h_q, invalid.size(-1))] = float("-inf")
    lse = aw.logsumexp(dim=-1)
    aw = torch.exp(aw - lse.unsqueeze(-1))
    out = aw @ gathered[..., :head_dim]
    out = out.view(b, s_q, h_q, head_dim)
    lse = lse.view(b, s_q, h_q)
    if attn_sink is not None:
        out *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(-1)
    lonely = lse == float("-inf")
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    return out.squeeze(1).to(torch.bfloat16)


def rocm_ref_sparse_attn_decode_flydsl(
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
    return rocm_ref_sparse_attn_decode_torch(
        q, blocked_k, indices_in_kvcache, topk_length, scale, head_dim, attn_sink,
        extra_blocked_k=extra_blocked_k,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
    )
