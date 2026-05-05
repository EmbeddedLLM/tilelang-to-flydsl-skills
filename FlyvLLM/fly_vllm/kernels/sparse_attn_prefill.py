"""sparse_attn_prefill.

Sparse attention with index-gathered KV.  Dominant compute is the
``q @ gathered_kv.transpose`` GEMM and the matmul with probabilities;
both lean on hipBLASLt.  A handcoded FlyDSL kernel would help only if
fused with the index_select + softmax (no temporary score tensor),
which is non-trivial.  Keeping the torch reference as the FlyDSL fast
path; benchmarks compare against torch + torch.compile.
"""

from __future__ import annotations

import torch


def rocm_ref_sparse_attn_prefill_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    indices = indices.clone().squeeze(1)
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    if topk_length is not None:
        mask = torch.arange(topk, device=indices.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid = (indices < 0) | (indices >= s_kv)
    indices[invalid] = 0
    qf = q.float()
    gathered = kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
    scores = qf @ gathered.transpose(1, 2)
    scores *= scale
    scores[invalid.unsqueeze(1).expand_as(scores)] = float("-inf")
    orig_lse = torch.logsumexp(scores, dim=-1)
    lse_for_o = orig_lse
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack(
                [orig_lse, attn_sink[:h_q].view(1, h_q).expand_as(orig_lse)], dim=0,
            ), dim=0,
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
    out = probs @ gathered[..., :head_dim]
    lonely = orig_lse == float("-inf")
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16)


def rocm_ref_sparse_attn_prefill_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    return rocm_ref_sparse_attn_prefill_torch(
        q, kv, indices, topk_length, scale, head_dim, attn_sink
    )
