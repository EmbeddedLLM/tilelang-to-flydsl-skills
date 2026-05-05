"""Correctness + benchmark for the inverse-RoPE pieces."""

from __future__ import annotations

import math

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.inv_rope import (
    apply_gptj_inv_rope_flydsl,
    apply_gptj_inv_rope_torch,
    rocm_inv_rope_einsum_flydsl,
    rocm_inv_rope_einsum_torch,
)


def _build_cos_sin_cache(max_pos: int, rope_dim: int) -> torch.Tensor:
    half = rope_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1).cuda()


@pytest.mark.parametrize("tokens", [1, 17, 128])
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("nope_dim,rope_dim", [(128, 64), (256, 64)])
def test_apply_gptj_inv_rope_correctness(tokens, groups, nope_dim, rope_dim):
    torch.manual_seed(0)
    head_dim = nope_dim + rope_dim
    cos_sin = _build_cos_sin_cache(2048, rope_dim)
    positions = torch.randint(0, 2048, (tokens,), dtype=torch.int32, device="cuda")
    if groups == 1:
        x = torch.randn(tokens, head_dim, dtype=torch.bfloat16, device="cuda")
    else:
        x = torch.randn(tokens, groups, head_dim, dtype=torch.bfloat16, device="cuda")
    out_ref = apply_gptj_inv_rope_torch(x, positions, cos_sin, rope_dim)
    out_fly = apply_gptj_inv_rope_flydsl(x, positions, cos_sin, rope_dim)
    torch.testing.assert_close(out_fly, out_ref)


@pytest.mark.parametrize("tokens,groups,nope_dim,rope_dim", [(512, 8, 256, 64)])
def test_apply_gptj_inv_rope_benchmark(tokens, groups, nope_dim, rope_dim, capsys):
    torch.manual_seed(0)
    head_dim = nope_dim + rope_dim
    cos_sin = _build_cos_sin_cache(2048, rope_dim)
    positions = torch.randint(0, 2048, (tokens,), dtype=torch.int32, device="cuda")
    x = torch.randn(tokens, groups, head_dim, dtype=torch.bfloat16, device="cuda")
    torch_compiled = torch.compile(apply_gptj_inv_rope_torch, dynamic=False)
    results = {
        "torch": bench(lambda: apply_gptj_inv_rope_torch(x, positions, cos_sin, rope_dim)),
        "compile": bench(lambda: torch_compiled(x, positions, cos_sin, rope_dim)),
        "flydsl": bench(lambda: apply_gptj_inv_rope_flydsl(x, positions, cos_sin, rope_dim)),
    }
    label = f"inv_rope t={tokens} g={groups} nd={nope_dim} rd={rope_dim}"
    with capsys.disabled():
        print()
        print(report(label, results))


class _DummyRotary(torch.nn.Module):
    def __init__(self, cos_sin: torch.Tensor):
        super().__init__()
        self.cos_sin_cache = cos_sin


class _DummyWoA(torch.nn.Module):
    def __init__(self, n_groups: int, lora: int, hidden_per_group: int):
        super().__init__()
        # Weight reshapes to (n_groups, lora, hidden_per_group) for the einsum.
        self.weight = torch.nn.Parameter(
            torch.randn(n_groups * lora, hidden_per_group, dtype=torch.bfloat16, device="cuda") * 0.02
        )


@pytest.mark.parametrize("tokens", [1, 33, 128])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("nope_dim,rope_dim", [(128, 64)])
@pytest.mark.parametrize("lora", [256])
def test_rocm_inv_rope_einsum_correctness(tokens, groups, nope_dim, rope_dim, lora):
    torch.manual_seed(0)
    head_dim = nope_dim + rope_dim
    cos_sin = _build_cos_sin_cache(2048, rope_dim)
    rot = _DummyRotary(cos_sin)
    wo_a = _DummyWoA(groups, lora, head_dim)
    positions = torch.randint(0, 2048, (tokens,), dtype=torch.int32, device="cuda")
    o = torch.randn(tokens, groups * head_dim, dtype=torch.bfloat16, device="cuda")
    out_ref = rocm_inv_rope_einsum_torch(rot, o, positions, rope_dim, groups, lora, wo_a)
    out_fly = rocm_inv_rope_einsum_flydsl(rot, o, positions, rope_dim, groups, lora, wo_a)
    torch.testing.assert_close(out_fly, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("tokens,groups,nope_dim,rope_dim,lora",
                         [(512, 8, 256, 64, 512)])
def test_rocm_inv_rope_einsum_benchmark(tokens, groups, nope_dim, rope_dim, lora, capsys):
    torch.manual_seed(0)
    head_dim = nope_dim + rope_dim
    cos_sin = _build_cos_sin_cache(2048, rope_dim)
    rot = _DummyRotary(cos_sin)
    wo_a = _DummyWoA(groups, lora, head_dim)
    positions = torch.randint(0, 2048, (tokens,), dtype=torch.int32, device="cuda")
    o = torch.randn(tokens, groups * head_dim, dtype=torch.bfloat16, device="cuda")

    torch_compiled = torch.compile(rocm_inv_rope_einsum_torch, dynamic=False)
    results = {
        "torch": bench(lambda: rocm_inv_rope_einsum_torch(rot, o, positions, rope_dim, groups, lora, wo_a)),
        "compile": bench(lambda: torch_compiled(rot, o, positions, rope_dim, groups, lora, wo_a)),
        "flydsl": bench(lambda: rocm_inv_rope_einsum_flydsl(rot, o, positions, rope_dim, groups, lora, wo_a)),
    }
    label = f"inv_rope_einsum t={tokens} g={groups} d={head_dim} lora={lora}"
    with capsys.disabled():
        print()
        print(report(label, results))
