"""Correctness + benchmark for the sparse-attention prefill / decode paths."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.sparse_attn_prefill import (
    rocm_ref_sparse_attn_prefill_flydsl,
    rocm_ref_sparse_attn_prefill_torch,
)
from fly_vllm.kernels.sparse_attn_decode import (
    rocm_ref_sparse_attn_decode_flydsl,
    rocm_ref_sparse_attn_decode_torch,
)


def _make_prefill_inputs(s_q: int, s_kv: int, h_q: int, d_qk: int, head_dim: int, topk: int):
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, d_qk, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, s_kv, (s_q, 1, topk), dtype=torch.int32, device=device)
    if s_q > 2 and topk > 4:
        indices[0, 0, 2] = -1  # exercise invalid sentinel
    topk_length = torch.randint(topk // 2, topk + 1, (s_q,), dtype=torch.int32, device=device)
    return q, kv, indices, topk_length


@pytest.mark.parametrize("s_q,s_kv,h_q,d_qk,head_dim,topk",
                         [(8, 128, 4, 192, 128, 32),
                          (33, 256, 8, 192, 128, 64),
                          # DeepSeek V4 absorbed-MLA shapes (small s_q for speed).
                          (8, 1024, 16, 576, 512, 512),
                          (8, 2048, 16, 576, 512, 1024)],
                         ids=["v3_topk32", "v3_topk64", "v4flash", "v4pro"])
def test_sparse_attn_prefill_correctness(s_q, s_kv, h_q, d_qk, head_dim, topk):
    q, kv, idx, lens = _make_prefill_inputs(s_q, s_kv, h_q, d_qk, head_dim, topk)
    scale = 1.0 / (d_qk ** 0.5)
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda") * 0.1
    out_ref = rocm_ref_sparse_attn_prefill_torch(q, kv, idx, lens, scale, head_dim, sink)
    out_fly = rocm_ref_sparse_attn_prefill_flydsl(q, kv, idx, lens, scale, head_dim, sink)
    torch.testing.assert_close(out_fly, out_ref, atol=1e-2, rtol=1e-2, equal_nan=True)


@pytest.mark.parametrize("s_q,s_kv,h_q,d_qk,head_dim,topk",
                         [
                             # original V3-style config
                             (64, 1024, 8, 192, 128, 256),
                             # DeepSeek V4 Flash, TP=4
                             (64, 2048, 16, 576, 512, 512),
                             # DeepSeek V4 Pro, TP=8
                             (64, 4096, 16, 576, 512, 1024),
                         ],
                         ids=["v3_h8_topk256", "v4flash_tp4", "v4pro_tp8"])
def test_sparse_attn_prefill_benchmark(s_q, s_kv, h_q, d_qk, head_dim, topk, capsys):
    q, kv, idx, lens = _make_prefill_inputs(s_q, s_kv, h_q, d_qk, head_dim, topk)
    scale = 1.0 / (d_qk ** 0.5)
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda") * 0.1
    # torch.compile spends >10s warming up at the larger DSv4 sizes — skip it
    # for the d_qk>=576 configs where the compile-warmup cost dominates the
    # benchmark wall-clock.
    results = {
        "torch": bench(lambda: rocm_ref_sparse_attn_prefill_torch(q, kv, idx, lens, scale, head_dim, sink)),
        "flydsl": bench(lambda: rocm_ref_sparse_attn_prefill_flydsl(q, kv, idx, lens, scale, head_dim, sink)),
    }
    if d_qk < 576:
        torch_compiled = torch.compile(rocm_ref_sparse_attn_prefill_torch, dynamic=False)
        results["compile"] = bench(lambda: torch_compiled(q, kv, idx, lens, scale, head_dim, sink))
    label = f"sparse_prefill q={s_q} kv={s_kv} h={h_q} d={d_qk} topk={topk}"
    with capsys.disabled():
        print()
        print(report(label, results))


def _make_decode_inputs(b: int, s_q: int, h_q: int, d_qk: int, s_kv: int, topk: int):
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=device)
    blocked_k = torch.randn(s_kv, d_qk, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, s_kv, (b, s_q, topk), dtype=torch.int32, device=device)
    if topk > 4:
        indices[0, 0, 2] = -1
    topk_length = torch.randint(topk // 2, topk + 1, (b,), dtype=torch.int32, device=device)
    return q, blocked_k, indices, topk_length


@pytest.mark.parametrize("b,s_q,h_q,d_qk,s_kv,topk",
                         [(2, 1, 4, 192, 128, 32),
                          (4, 2, 8, 192, 256, 64)])
def test_sparse_attn_decode_correctness(b, s_q, h_q, d_qk, s_kv, topk):
    q, kv, idx, lens = _make_decode_inputs(b, s_q, h_q, d_qk, s_kv, topk)
    scale = 1.0 / (d_qk ** 0.5)
    head_dim = 128
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda") * 0.1
    out_ref = rocm_ref_sparse_attn_decode_torch(q, kv, idx, lens, scale, head_dim, sink)
    out_fly = rocm_ref_sparse_attn_decode_flydsl(q, kv, idx, lens, scale, head_dim, sink)
    torch.testing.assert_close(out_fly, out_ref, atol=1e-2, rtol=1e-2, equal_nan=True)


@pytest.mark.parametrize("b,s_q,h_q,d_qk,s_kv,topk",
                         [(8, 1, 8, 192, 1024, 256)])
def test_sparse_attn_decode_benchmark(b, s_q, h_q, d_qk, s_kv, topk, capsys):
    q, kv, idx, lens = _make_decode_inputs(b, s_q, h_q, d_qk, s_kv, topk)
    scale = 1.0 / (d_qk ** 0.5)
    head_dim = 128
    sink = torch.randn(h_q, dtype=torch.float32, device="cuda") * 0.1
    torch_compiled = torch.compile(rocm_ref_sparse_attn_decode_torch, dynamic=False)
    results = {
        "torch": bench(lambda: rocm_ref_sparse_attn_decode_torch(q, kv, idx, lens, scale, head_dim, sink)),
        "compile": bench(lambda: torch_compiled(q, kv, idx, lens, scale, head_dim, sink)),
        "flydsl": bench(lambda: rocm_ref_sparse_attn_decode_flydsl(q, kv, idx, lens, scale, head_dim, sink)),
    }
    label = f"sparse_decode b={b} h={h_q} d={d_qk} kv={s_kv} topk={topk}"
    with capsys.disabled():
        print()
        print(report(label, results))
