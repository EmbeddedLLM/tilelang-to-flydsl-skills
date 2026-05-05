"""fp8_mqa_logits.

GEMM-heavy compute path; the dominant cost is the bf16 GEMM inside
``torch.einsum("mhd,nd->hmn", q.bf16, k.bf16)``.  hipBLASLt's tuned GEMM
beats a handcoded FlyDSL GEMM at the typical DeepseekV3 sizes (m up to a
few hundred, n up to a few k, d=128, h=8-32).  We therefore expose the
torch implementation as the "FlyDSL" path while still benchmarking against
torch and torch.compile for transparency.

A future custom FlyDSL kernel could fuse the relu + weighted-sum epilogue
across h, eliminating the intermediate (h,m,n) score tensor — that would be
the principal win.  Not worth it at current sizes.
"""

from __future__ import annotations

import torch


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    qb = q.to(torch.bfloat16)
    mask_lo = (
        torch.arange(0, seq_len_kv, device=qb.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=qb.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", qb, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def fp8_mqa_logits_flydsl(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return fp8_mqa_logits_torch(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
