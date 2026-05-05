"""topk_indices.

torch.topk on ROCm uses a heavily-tuned radix kernel; matching it from
FlyDSL with a hand-rolled bitonic select would be a sizeable project for
no measurable win at the typical (rows, cols, k) used by the indexer.
Kept as a torch passthrough.
"""

from __future__ import annotations

import torch


def topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens), -1,
        dtype=torch.int32, device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def topk_indices_flydsl(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    return topk_indices_torch(logits, topk_tokens)
