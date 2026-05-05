"""Correctness + benchmark for rocm_dequantize_blocked_k_cache."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.dequantize_blocked_k_cache import (
    rocm_dequantize_blocked_k_cache_flydsl,
    rocm_dequantize_blocked_k_cache_torch,
)


def _make_cache(num_blocks: int, block_size: int, nope_head_dim: int, rope_head_dim: int):
    """Build a (num_blocks, block_size, head_dim + 4) uint8 cache.

    The upstream dequant treats the per-block bytes as block-major: first all
    positions' nope+rope contiguously, then all positions' 8-byte scale rows.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_tiles = nope_head_dim // 64
    nope_rope = nope_head_dim + 2 * rope_head_dim
    packed = nope_rope + 8  # last-dim allocation per (block, pos)
    cache = torch.empty(num_blocks, block_size, packed, dtype=torch.uint8, device=device)
    flat = cache.view(num_blocks, -1)

    # nope+rope: block-major contiguous
    nope_f = torch.randn(num_blocks, block_size, nope_head_dim, device=device) * 2.0
    nope_fp8 = nope_f.to(torch.float8_e4m3fn).view(torch.uint8)
    rope_bf = torch.randn(num_blocks, block_size, rope_head_dim, device=device).to(torch.bfloat16)

    nr_view = flat[:, : block_size * nope_rope].view(num_blocks, block_size, nope_rope)
    nr_view[:, :, :nope_head_dim] = nope_fp8
    nr_view[:, :, nope_head_dim:] = rope_bf.view(torch.uint8)

    # scale: e8m0 bytes in [120, 134] -- 2^-7 .. 2^7
    sc_byte = torch.randint(120, 134, (num_blocks, block_size, num_tiles),
                            dtype=torch.uint8, device=device)
    sc_view = flat[:, block_size * nope_rope :].view(num_blocks, block_size, 8)
    sc_view[:, :, :num_tiles] = sc_byte
    return cache


@pytest.mark.parametrize("num_blocks", [4, 16])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("nope_head_dim,rope_head_dim,head_dim",
                         [(128, 64, 192), (256, 64, 320), (512, 64, 576)])
def test_dequantize_blocked_k_cache_correctness(
    num_blocks, block_size, nope_head_dim, rope_head_dim, head_dim,
):
    cache = _make_cache(num_blocks, block_size, nope_head_dim, rope_head_dim)
    out_ref = rocm_dequantize_blocked_k_cache_torch(
        cache, head_dim=head_dim,
        nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
    )
    out_fly = rocm_dequantize_blocked_k_cache_flydsl(
        cache, head_dim=head_dim,
        nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
    )
    torch.testing.assert_close(out_fly, out_ref)


@pytest.mark.parametrize("num_blocks,block_size,nope_head_dim,rope_head_dim,head_dim",
                         [(64, 64, 512, 64, 576)])
def test_dequantize_blocked_k_cache_benchmark(
    num_blocks, block_size, nope_head_dim, rope_head_dim, head_dim, capsys,
):
    cache = _make_cache(num_blocks, block_size, nope_head_dim, rope_head_dim)

    def torch_fn():
        return rocm_dequantize_blocked_k_cache_torch(
            cache, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
        )

    torch_compiled = torch.compile(rocm_dequantize_blocked_k_cache_torch, dynamic=False)

    def compile_fn():
        return torch_compiled(
            cache, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
        )

    def fly_fn():
        return rocm_dequantize_blocked_k_cache_flydsl(
            cache, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
        )

    results = {
        "torch": bench(torch_fn),
        "compile": bench(compile_fn),
        "flydsl": bench(fly_fn),
    }
    label = f"dequant nb={num_blocks} bs={block_size} hd={head_dim}"
    with capsys.disabled():
        print()
        print(report(label, results))
