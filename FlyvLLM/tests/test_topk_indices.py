"""Correctness + benchmark for topk_indices."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.topk_indices import topk_indices_flydsl, topk_indices_torch


@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("cols", [128, 1024])
@pytest.mark.parametrize("topk", [16, 64, 256])
def test_topk_indices_correctness(rows, cols, topk):
    torch.manual_seed(0)
    logits = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    # inject some -inf to exercise the sentinel path
    if cols >= 4:
        logits[..., : cols // 4] = float("-inf")
    out_ref = topk_indices_torch(logits, topk)
    out_fly = topk_indices_flydsl(logits, topk)
    # ordering of topk values may differ when ties exist; just compare the *set*.
    for r in range(rows):
        ref_set = set(out_ref[r].tolist())
        fly_set = set(out_fly[r].tolist())
        assert ref_set == fly_set, f"row {r} sets differ"


@pytest.mark.parametrize("rows,cols,topk", [(64, 4096, 256)])
def test_topk_indices_benchmark(rows, cols, topk, capsys):
    torch.manual_seed(0)
    logits = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
    logits[..., : cols // 4] = float("-inf")

    torch_compiled = torch.compile(topk_indices_torch, dynamic=False)
    results = {
        "torch": bench(lambda: topk_indices_torch(logits, topk)),
        "compile": bench(lambda: torch_compiled(logits, topk)),
        "flydsl": bench(lambda: topk_indices_flydsl(logits, topk)),
    }
    label = f"topk r={rows} c={cols} k={topk}"
    with capsys.disabled():
        print()
        print(report(label, results))
