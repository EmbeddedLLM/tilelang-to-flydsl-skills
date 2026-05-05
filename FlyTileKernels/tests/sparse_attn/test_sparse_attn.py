"""Correctness tests mirroring the tilelang-ascend sparse_flash_attention example.

We import the upstream torch reference (``sparse_attn`` in
``tilelang-ascend/examples/deepseek_v4/sparse_flash_attention.py``) by
re-implementing the same functions verbatim here, so the FlyTileKernels
port is validated against the *exact* mathematical specification the
upstream test asserts on.

The upstream test:

    b, m, n, h, d, topk = 1, 256, 256, 64, 512, 128
    softmax_scale = 512**-0.5
    rtol = atol = 1e-2

This file replicates that with the same tolerance plus the second
("Shape 2") config that the upstream file documents in a commented-out
line.
"""

from __future__ import annotations

import pytest
import torch

from fly_tile_kernels.sparse_attn import sparse_attn


# ---------------------------------------------------------------------------
# Golden — verbatim from tilelang-ascend/examples/deepseek_v4/sparse_flash_attention.py
# ---------------------------------------------------------------------------

def gather_sparse_kv(kv_states: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, topk = topk_idxs.shape
    batch_idx = (
        torch.arange(batch_size, device=kv_states.device)
        .view(batch_size, 1, 1)
        .expand(-1, seq_len, topk)
    )
    safe_topk_idxs = torch.where(topk_idxs == -1, 0, topk_idxs).long()
    gathered = kv_states[batch_idx, safe_topk_idxs, :]
    gathered_mask = (topk_idxs != -1).unsqueeze(-1).to(gathered.dtype)
    return gathered * gathered_mask


def sparse_softmax_with_sink(
    scores: torch.Tensor,
    attn_sink: torch.Tensor,
    head_dim: int,
    softmax_dim: int = -1,
) -> torch.Tensor:
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
    sparse_kv = gather_sparse_kv(kv_states, topk_idxs)
    score_mask = torch.where(
        (topk_idxs == -1).unsqueeze(-2),
        -torch.inf,
        0.0,
    ).to(device=query_states.device, dtype=torch.float32)
    scores = torch.matmul(query_states, sparse_kv.transpose(-2, -1)).to(torch.float32)
    probs = sparse_softmax_with_sink(
        scores * softmax_scale + score_mask, attn_sink, head_dim=-2,
    )
    return torch.matmul(probs, sparse_kv.to(torch.float32))


def make_random_test_inputs(b, m, n, h, d, topk, dtype):
    torch.manual_seed(42)
    return {
        "q": torch.randn((b, m, h, d), dtype=dtype, device="cuda"),
        "kv": torch.randn((b, n, d), dtype=dtype, device="cuda"),
        "attn_sink": torch.randn((h,), dtype=torch.float32, device="cuda"),
        "topk_idxs": torch.randint(0, n, (b, m, topk), dtype=torch.int32, device="cuda"),
    }


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "b,m,n,h,d,topk",
    [
        # Upstream Ascend test "Shape 1" — the configuration assert_close'd
        # in tilelang-ascend's test() function.
        (1, 256, 256, 64, 512, 128),
        # Upstream Ascend test "Shape 2" — left as a commented alternative
        # in the upstream file; we exercise it to catch shape-handling bugs.
        # The upstream uses topk=6 here, but our kernel requires topk%32==0,
        # so we round to topk=64 (still tiny).
        (1, 6, 64, 16, 512, 64),
    ],
    ids=["ascend_shape1", "ascend_shape2"],
)
def test_sparse_attn_matches_torch_ref(b, m, n, h, d, topk):
    """Same tolerance the upstream Ascend test uses (rtol=atol=1e-2)."""
    inputs = make_random_test_inputs(b, m, n, h, d, topk, torch.bfloat16)
    softmax_scale = 512.0 ** -0.5

    out_ref = sparse_attn_torch(
        inputs["q"], inputs["kv"], inputs["attn_sink"],
        inputs["topk_idxs"], softmax_scale,
    )
    out_fly = sparse_attn(
        inputs["q"], inputs["kv"], inputs["attn_sink"],
        inputs["topk_idxs"], softmax_scale,
    )

    assert out_ref.shape == (b, m, h, d)
    assert out_fly.shape == (b, m, h, d)
    assert out_ref.dtype == torch.float32
    assert out_fly.dtype == torch.float32

    torch.testing.assert_close(out_fly, out_ref, rtol=1e-2, atol=1e-2)


def test_sparse_attn_handles_invalid_indices():
    """A few -1 indices in the topk batch should match the upstream gather
    behaviour (the corresponding KV positions contribute zero rows)."""
    b, m, n, h, d, topk = 1, 16, 256, 16, 512, 64
    inputs = make_random_test_inputs(b, m, n, h, d, topk, torch.bfloat16)
    # Poke a few -1's.
    inputs["topk_idxs"][0, 0, 0] = -1
    inputs["topk_idxs"][0, 5, 3] = -1
    inputs["topk_idxs"][0, m - 1, topk - 1] = -1
    softmax_scale = 512.0 ** -0.5

    out_ref = sparse_attn_torch(
        inputs["q"], inputs["kv"], inputs["attn_sink"],
        inputs["topk_idxs"], softmax_scale,
    )
    out_fly = sparse_attn(
        inputs["q"], inputs["kv"], inputs["attn_sink"],
        inputs["topk_idxs"], softmax_scale,
    )
    torch.testing.assert_close(out_fly, out_ref, rtol=1e-2, atol=1e-2)
