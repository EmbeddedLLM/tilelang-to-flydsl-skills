"""Correctness + benchmark for fp8_mqa_logits."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.fp8_mqa_logits import (
    fp8_mqa_logits_flydsl,
    fp8_mqa_logits_torch,
)


def _make_inputs(m: int, n: int, h: int, d: int):
    torch.manual_seed(0)
    device = "cuda"
    q = (torch.randn(m, h, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(n, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.randn(n, device=device).abs() * 0.5 + 0.1
    weights = torch.randn(m, h, device=device) * 0.5
    cu_seqlen_ks = torch.zeros(m, dtype=torch.int32, device=device)
    cu_seqlen_ke = torch.full((m,), n, dtype=torch.int32, device=device)
    return q, k, scale, weights, cu_seqlen_ks, cu_seqlen_ke


@pytest.mark.parametrize("m,n,h,d", [(8, 128, 4, 128), (32, 256, 8, 128)])
def test_fp8_mqa_logits_correctness(m, n, h, d):
    q, k, scale, w, ks, ke = _make_inputs(m, n, h, d)
    out_ref = fp8_mqa_logits_torch(q, k, scale, w, ks, ke)
    out_fly = fp8_mqa_logits_flydsl(q, k, scale, w, ks, ke)
    torch.testing.assert_close(out_fly, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("m,n,h,d", [(64, 1024, 8, 128)])
def test_fp8_mqa_logits_benchmark(m, n, h, d, capsys):
    q, k, scale, w, ks, ke = _make_inputs(m, n, h, d)
    torch_compiled = torch.compile(fp8_mqa_logits_torch, dynamic=False)
    results = {
        "torch": bench(lambda: fp8_mqa_logits_torch(q, k, scale, w, ks, ke)),
        "compile": bench(lambda: torch_compiled(q, k, scale, w, ks, ke)),
        "flydsl": bench(lambda: fp8_mqa_logits_flydsl(q, k, scale, w, ks, ke)),
    }
    label = f"fp8_mqa m={m} n={n} h={h} d={d}"
    with capsys.disabled():
        print()
        print(report(label, results))
