"""Correctness + benchmark for fp8_paged_mqa_logits."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits_flydsl,
    fp8_paged_mqa_logits_torch,
)


def _make_paged_inputs(b, next_n, h, d, num_blocks, block_size, max_model_len):
    torch.manual_seed(0)
    device = "cuda"
    q = (torch.randn(b, next_n, h, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    # KV cache: (num_blocks, block_size, 1, d + 4) uint8 packed; scale fp32 in last 4 bytes
    cache = torch.zeros(num_blocks, block_size, 1, d + 4, dtype=torch.uint8, device=device)
    val = (torch.randn(num_blocks, block_size, 1, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    scale = (torch.randn(num_blocks, block_size, 1, device=device).abs() * 0.5 + 0.1).to(torch.float32)
    cache[..., :d] = val.view(torch.uint8)
    cache[..., d:] = scale.view(torch.uint8).view(num_blocks, block_size, 1, 4)
    weights = (torch.randn(b * next_n, h, device=device) * 0.5).to(torch.float32)
    context_lens = torch.randint(8, block_size * 2, (b,), dtype=torch.int32, device=device)
    if next_n > 1:
        # ensure ctx >= next_n so q_offsets are nonneg
        context_lens.clamp_(min=next_n)
    max_blocks = (max_model_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (b, max_blocks),
                                 dtype=torch.int32, device=device)
    return q, cache, weights, context_lens, block_tables


@pytest.mark.parametrize("b,next_n,h,d,num_blocks,block_size,max_model_len",
                         [(2, 1, 4, 128, 16, 64, 256),
                          (2, 2, 4, 128, 16, 64, 256)])
def test_fp8_paged_mqa_logits_correctness(b, next_n, h, d, num_blocks, block_size, max_model_len):
    q, cache, w, ctx, bt = _make_paged_inputs(b, next_n, h, d, num_blocks, block_size, max_model_len)
    out_ref = fp8_paged_mqa_logits_torch(q, cache, w, ctx, bt, max_model_len)
    out_fly = fp8_paged_mqa_logits_flydsl(q, cache, w, ctx, bt, max_model_len)
    torch.testing.assert_close(out_fly, out_ref, atol=2e-1, rtol=2e-1, equal_nan=True)


@pytest.mark.parametrize("b,next_n,h,d,num_blocks,block_size,max_model_len",
                         [(8, 1, 8, 128, 64, 64, 1024)])
def test_fp8_paged_mqa_logits_benchmark(b, next_n, h, d, num_blocks, block_size, max_model_len, capsys):
    q, cache, w, ctx, bt = _make_paged_inputs(b, next_n, h, d, num_blocks, block_size, max_model_len)
    torch_compiled = torch.compile(fp8_paged_mqa_logits_torch, dynamic=False)
    results = {
        "torch": bench(lambda: fp8_paged_mqa_logits_torch(q, cache, w, ctx, bt, max_model_len)),
        "compile": bench(lambda: torch_compiled(q, cache, w, ctx, bt, max_model_len)),
        "flydsl": bench(lambda: fp8_paged_mqa_logits_flydsl(q, cache, w, ctx, bt, max_model_len)),
    }
    label = f"fp8_paged b={b} h={h} d={d} ctx={max_model_len}"
    with capsys.disabled():
        print()
        print(report(label, results))
