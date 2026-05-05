"""Correctness + benchmark for indexer_k_quant_and_cache + cp_gather_indexer."""

from __future__ import annotations

import pytest
import torch

from fly_vllm.kernels._bench import bench, report
from fly_vllm.kernels.indexer_k_quant_and_cache import (
    indexer_k_quant_and_cache_flydsl,
    indexer_k_quant_and_cache_torch,
)
from fly_vllm.kernels.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache_flydsl,
    cp_gather_indexer_k_quant_cache_torch,
)


def _make_cache(num_blocks: int, block_size: int, head_dim: int):
    """Allocate (num_blocks, block_size, head_dim + 4) uint8 cache.

    Layout per block (in flat byte order): first BS*head_dim bytes are FP8
    K values (any layout — the kernel writes in SHUFFLE order); next BS*4 bytes
    are float32 scales.
    """
    return torch.zeros(num_blocks, block_size, head_dim + 4, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("num_tokens", [1, 17, 64])
@pytest.mark.parametrize("num_blocks,block_size", [(8, 64)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_indexer_k_quant_and_cache_correctness(num_tokens, num_blocks, block_size, head_dim, scale_fmt):
    torch.manual_seed(0)
    device = "cuda"
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 2.0
    slot = torch.randint(0, num_blocks * block_size, (num_tokens,),
                         dtype=torch.int32, device=device)
    if num_tokens > 1:
        slot[0] = -1  # exercise skip path

    cache_ref = _make_cache(num_blocks, block_size, head_dim)
    cache_fly = _make_cache(num_blocks, block_size, head_dim)
    indexer_k_quant_and_cache_torch(k, cache_ref, slot, 128, scale_fmt)
    indexer_k_quant_and_cache_flydsl(k, cache_fly, slot, 128, scale_fmt)
    # FP8 quantisation: allow off-by-1 in the FP8 byte (1 ULP at the rounding
    # boundary).  The 4 fp32 scale bytes per (block, pos) must match exactly.
    flat_ref = cache_ref.view(num_blocks, -1)
    flat_fly = cache_fly.view(num_blocks, -1)
    val_ref = flat_ref[:, : block_size * head_dim].to(torch.int32)
    val_fly = flat_fly[:, : block_size * head_dim].to(torch.int32)
    diff = (val_ref - val_fly).abs()
    n_off = (diff > 1).sum().item()
    n_off_eq1 = ((diff == 1).sum().item())
    total = val_ref.numel()
    assert n_off == 0, f"{n_off}/{total} fp8 bytes differ by >1 ULP"
    assert n_off_eq1 / total < 1e-3, f"too many 1-ULP fp8 mismatches: {n_off_eq1}/{total}"
    scale_ref = flat_ref[:, block_size * head_dim :]
    scale_fly = flat_fly[:, block_size * head_dim :]
    torch.testing.assert_close(scale_fly, scale_ref)


@pytest.mark.parametrize("num_tokens,num_blocks,block_size,head_dim",
                         [(512, 64, 64, 128)])
def test_indexer_k_quant_and_cache_benchmark(num_tokens, num_blocks, block_size, head_dim, capsys):
    torch.manual_seed(0)
    device = "cuda"
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 2.0
    slot = torch.randint(0, num_blocks * block_size, (num_tokens,),
                         dtype=torch.int32, device=device)

    cache_t = _make_cache(num_blocks, block_size, head_dim)
    cache_f = _make_cache(num_blocks, block_size, head_dim)

    def torch_fn():
        cache_t.zero_()
        indexer_k_quant_and_cache_torch(k, cache_t, slot, 128, None)

    def fly_fn():
        cache_f.zero_()
        indexer_k_quant_and_cache_flydsl(k, cache_f, slot, 128, None)

    results = {
        "torch": bench(torch_fn),
        "flydsl": bench(fly_fn),
    }
    label = f"k_quant_cache nt={num_tokens} nb={num_blocks} d={head_dim}"
    with capsys.disabled():
        print()
        print(report(label, results))


def _seed_cache(num_blocks: int, block_size: int, head_dim: int):
    """Build a cache by quantising random K via the torch reference."""
    torch.manual_seed(0)
    device = "cuda"
    cache = _make_cache(num_blocks, block_size, head_dim)
    num_tokens = num_blocks * block_size
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 2.0
    slot = torch.arange(num_tokens, dtype=torch.int32, device=device)
    indexer_k_quant_and_cache_torch(k, cache, slot, 128, None)
    return cache, k


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen", [33, 80])
@pytest.mark.parametrize("num_blocks,block_size,head_dim", [(8, 64, 128)])
def test_cp_gather_indexer_correctness(batch_size, seqlen, num_blocks, block_size, head_dim):
    torch.manual_seed(0)
    device = "cuda"
    cache, _ = _seed_cache(num_blocks, block_size, head_dim)

    cu_seqlen = torch.tensor([0, seqlen, 2 * seqlen][:batch_size + 1], dtype=torch.int32, device=device)
    total = int(cu_seqlen[-1].item())
    token_to_seq = torch.zeros(total, dtype=torch.int32, device=device)
    for i in range(batch_size):
        token_to_seq[int(cu_seqlen[i].item()) : int(cu_seqlen[i + 1].item())] = i
    max_blocks = (seqlen + block_size - 1) // block_size
    block_table = torch.randint(0, num_blocks, (batch_size, max_blocks),
                                dtype=torch.int32, device=device)

    k_fp8_ref = torch.zeros(total, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k_scale_ref = torch.zeros(total, 4, dtype=torch.uint8, device=device)
    k_fp8_fly = torch.zeros_like(k_fp8_ref)
    k_scale_fly = torch.zeros_like(k_scale_ref)

    cp_gather_indexer_k_quant_cache_torch(
        cache, k_fp8_ref, k_scale_ref, block_table, cu_seqlen, token_to_seq,
    )
    cp_gather_indexer_k_quant_cache_flydsl(
        cache, k_fp8_fly, k_scale_fly, block_table, cu_seqlen, token_to_seq,
    )
    torch.testing.assert_close(k_fp8_fly.view(torch.uint8), k_fp8_ref.view(torch.uint8))
    torch.testing.assert_close(k_scale_fly, k_scale_ref)


@pytest.mark.parametrize("total_tokens,num_blocks,block_size,head_dim",
                         [(512, 64, 64, 128)])
def test_cp_gather_indexer_benchmark(total_tokens, num_blocks, block_size, head_dim, capsys):
    torch.manual_seed(0)
    device = "cuda"
    cache, _ = _seed_cache(num_blocks, block_size, head_dim)
    bs = 4
    seqlen = total_tokens // bs
    cu_seqlen = torch.arange(0, total_tokens + 1, seqlen, dtype=torch.int32, device=device)
    token_to_seq = torch.zeros(total_tokens, dtype=torch.int32, device=device)
    for i in range(bs):
        token_to_seq[i * seqlen : (i + 1) * seqlen] = i
    max_blocks = (seqlen + block_size - 1) // block_size
    block_table = torch.randint(0, num_blocks, (bs, max_blocks),
                                dtype=torch.int32, device=device)

    k_fp8 = torch.zeros(total_tokens, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.zeros(total_tokens, 4, dtype=torch.uint8, device=device)

    def torch_fn():
        cp_gather_indexer_k_quant_cache_torch(
            cache, k_fp8, k_scale, block_table, cu_seqlen, token_to_seq,
        )

    def fly_fn():
        cp_gather_indexer_k_quant_cache_flydsl(
            cache, k_fp8, k_scale, block_table, cu_seqlen, token_to_seq,
        )

    results = {
        "torch": bench(torch_fn),
        "flydsl": bench(fly_fn),
    }
    label = f"cp_gather t={total_tokens} nb={num_blocks} d={head_dim}"
    with capsys.disabled():
        print()
        print(report(label, results))
