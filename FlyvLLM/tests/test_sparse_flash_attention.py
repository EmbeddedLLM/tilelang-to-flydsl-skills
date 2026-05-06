"""Correctness tests for the port of tilelang-ascend's sparse_flash_attention.

Goldens:
- ``sparse_attn_torch``        — direct port of the upstream torch reference
- ``sparse_attn`` (= flydsl)    — fly_vllm fast path; falls back to torch on
                                  shapes the FlyDSL prefill kernel can't handle
"""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.sparse_flash_attention import (
    gather_sparse_kv,
    sparse_attn,
    sparse_attn_torch,
    sparse_softmax_with_sink,
)


def _make_inputs(b, m, n, h, d, topk, dtype=torch.bfloat16):
    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn((b, m, h, d), dtype=dtype, device=device)
    kv = torch.randn((b, n, d), dtype=dtype, device=device)
    attn_sink = torch.randn((h,), dtype=torch.float32, device=device)
    topk_idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32, device=device)
    # exercise the -1 mask: poke a few entries
    if topk >= 4:
        topk_idxs[0, 0, 0] = -1
        topk_idxs[0, m - 1, topk - 1] = -1
    return q, kv, attn_sink, topk_idxs


# ---------------------------------------------------------------------------
# Helper-level smoke tests (mirror the upstream gather + softmax helpers).
# ---------------------------------------------------------------------------

def test_gather_sparse_kv_zeroes_invalid():
    kv = torch.arange(2 * 4 * 3, dtype=torch.float32, device="cuda").reshape(2, 4, 3)
    idx = torch.tensor([[[0, 2, -1]], [[1, -1, 3]]], dtype=torch.int32, device="cuda")
    out = gather_sparse_kv(kv, idx)
    assert out.shape == (2, 1, 3, 3)
    # batch 0 row 0 = kv[0, 0]; row 1 = kv[0, 2]; row 2 = zeros (idx==-1)
    torch.testing.assert_close(out[0, 0, 0], kv[0, 0])
    torch.testing.assert_close(out[0, 0, 2], torch.zeros(3, device="cuda"))
    torch.testing.assert_close(out[1, 0, 1], torch.zeros(3, device="cuda"))


def test_softmax_sink_reduces_to_softmax_when_sink_is_minus_inf():
    scores = torch.randn(1, 4, 8, dtype=torch.float32, device="cuda")
    sink = torch.full((4,), -1e9, dtype=torch.float32, device="cuda")  # ~ -inf
    out = sparse_softmax_with_sink(scores, sink, head_dim=-2)
    expected = torch.softmax(scores, dim=-1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Full-pipeline correctness.  The DeepseekV4 test config (d=512, h=64) lands
# on the torch fallback path; we still validate that ``sparse_attn`` and
# ``sparse_attn_torch`` agree under the same upstream tolerances.  Smaller
# configs hit the FlyDSL kernel path.
# ---------------------------------------------------------------------------

# A small set covering both code paths.
@pytest.mark.parametrize(
    "b,m,n,h,d,topk",
    [
        # Falls through to flydsl FA2 kernel: d ≤ 256, divisible.
        (1, 8, 128, 4, 128, 64),
        (2, 16, 256, 8, 128, 64),
        # Upstream "Shape 1" — falls back to torch because d=512.
        (1, 256, 256, 64, 512, 128),
        # Upstream "Shape 2" — also d=512, smaller.
        (1, 6, 64, 16, 512, 64),
    ],
    ids=["small_d128_t64", "small_d128_t64_b2", "ascend_shape1", "ascend_shape2"],
)
def test_sparse_attn_correctness(b, m, n, h, d, topk):
    q, kv, attn_sink, topk_idxs = _make_inputs(b, m, n, h, d, topk)
    softmax_scale = float(d) ** -0.5

    out_ref = sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale)
    out_fly = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    assert out_ref.dtype == torch.float32, "upstream output must be fp32"
    assert out_fly.dtype == torch.float32, "port must preserve fp32 output"
    assert out_ref.shape == (b, m, h, d)
    assert out_fly.shape == (b, m, h, d)

    # Same tolerance the upstream Ascend test uses: rtol=atol=1e-2.
    torch.testing.assert_close(out_fly, out_ref, atol=1e-2, rtol=1e-2)


def test_sparse_attn_handles_all_invalid():
    """All -1 indices: every query is "lonely".  The upstream torch
    reference produces NaN (max == -inf, exp(-inf - -inf) = NaN); the
    FlyDSL fast path explicitly forces a 0 output for lonely queries.
    Both behaviours are defensible — we just verify the kernel doesn't
    crash and returns the right shape/dtype on this edge case."""
    b, m, n, h, d, topk = 1, 4, 32, 4, 128, 64
    q, kv, attn_sink, topk_idxs = _make_inputs(b, m, n, h, d, topk)
    topk_idxs.fill_(-1)
    softmax_scale = float(d) ** -0.5
    out_ref = sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale)
    out_fly = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
    assert out_ref.shape == (b, m, h, d)
    assert out_fly.shape == (b, m, h, d)
    assert out_ref.dtype == torch.float32 and out_fly.dtype == torch.float32


# ---------------------------------------------------------------------------
# Benchmark on the upstream "Shape 1" config (b=1, m=256, n=256, h=64,
# d=512, topk=128).  This is the torch-fallback path on gfx950.  Skip
# torch.compile because the d=512 path warms up very slowly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "b,m,n,h,d,topk",
    [(1, 256, 256, 64, 512, 128)],
    ids=["ascend_shape1_bench"],
)
def test_sparse_attn_benchmark(b, m, n, h, d, topk, capsys):
    q, kv, attn_sink, topk_idxs = _make_inputs(b, m, n, h, d, topk)
    softmax_scale = float(d) ** -0.5

    results = {
        "torch": bench(lambda: sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale)),
        "flydsl": bench(lambda: sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)),
    }
    label = f"sparse_flash_attn b={b} m={m} h={h} d={d} topk={topk}"
    with capsys.disabled():
        print()
        print(report(label, results))
