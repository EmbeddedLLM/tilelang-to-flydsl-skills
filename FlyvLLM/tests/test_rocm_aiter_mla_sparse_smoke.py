"""Smoke test for the public ``fly_vllm.rocm_aiter_mla_sparse`` API.

Imports each public function and exercises the end-to-end path with tiny
inputs; correctness of each piece is covered by the per-kernel test files.
"""

from __future__ import annotations

import torch

from fly_vllm import rocm_aiter_mla_sparse as m


def test_public_api_imports():
    expected = [
        "rocm_dequantize_blocked_k_cache",
        "indexer_k_quant_and_cache",
        "cp_gather_indexer_k_quant_cache",
        "rocm_fp8_mqa_logits",
        "rocm_fp8_paged_mqa_logits",
        "topk_indices",
        "rocm_inv_rope_einsum",
        "rocm_ref_sparse_attn_prefill",
        "rocm_sparse_attn_prefill",
        "rocm_ref_sparse_attn_decode",
        "rocm_forward_decode_fallback",
        "_decode_e8m0_scales",
        "_expand_2d_block_scales",
    ]
    for name in expected:
        assert hasattr(m, name), f"missing public symbol {name}"


def test_helpers_e8m0():
    # E8M0 byte 127 is 2^0 = 1.0; byte 128 is 2.0; byte 126 is 0.5.
    raw = torch.tensor([126, 127, 128], dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    out = m._decode_e8m0_scales(raw)
    torch.testing.assert_close(out, torch.tensor([0.5, 1.0, 2.0], device="cuda"))


def test_helpers_expand_block_scales():
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    out = m._expand_2d_block_scales(s, rows=4, cols=4)
    expected = torch.tensor(
        [[1.0, 1.0, 2.0, 2.0],
         [1.0, 1.0, 2.0, 2.0],
         [3.0, 3.0, 4.0, 4.0],
         [3.0, 3.0, 4.0, 4.0]],
        device="cuda",
    )
    torch.testing.assert_close(out, expected)


def test_pipeline_quant_then_gather():
    """quant K → cache → gather back; bytes round-trip."""
    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, head_dim = 4, 64, 128
    bs = 2
    seq = 80
    total = bs * seq
    cache = torch.zeros(num_blocks, block_size, head_dim + 4, dtype=torch.uint8, device=device)
    k = torch.randn(total, head_dim, dtype=torch.bfloat16, device=device) * 2.0
    slot = torch.arange(total, dtype=torch.int32, device=device)
    m.indexer_k_quant_and_cache(k, cache, slot, 128, scale_fmt=None)

    cu_seqlen = torch.tensor([0, seq, 2 * seq], dtype=torch.int32, device=device)
    token_to_seq = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32, device=device),
        torch.tensor([seq] * bs, dtype=torch.int32, device=device),
    )
    max_blocks = (seq + block_size - 1) // block_size
    block_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)
    # populate block table: tokens 0..seq-1 → blocks 0..max_blocks-1; for batch1 use later blocks.
    for b in range(bs):
        for i in range(max_blocks):
            block_table[b, i] = b * max_blocks + i
    # Re-quant K such that slot mapping respects the block table.
    cache.zero_()
    new_slot = torch.empty(total, dtype=torch.int32, device=device)
    for t in range(total):
        b = t // seq
        offset = t % seq
        page = offset // block_size
        in_page = offset % block_size
        new_slot[t] = block_table[b, page] * block_size + in_page
    m.indexer_k_quant_and_cache(k, cache, new_slot, 128, scale_fmt=None)

    k_fp8 = torch.zeros(total, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.zeros(total, 4, dtype=torch.uint8, device=device)
    m.cp_gather_indexer_k_quant_cache(
        cache, k_fp8, k_scale, block_table, cu_seqlen, token_to_seq,
    )
    # Round-trip the fp8 + scale: dequant k_fp8 * scale and check it tracks
    # the original input.
    scale_f = k_scale.view(torch.float32)
    k_back = k_fp8.to(torch.float32) * scale_f.unsqueeze(-1)
    diff = (k_back - k.to(torch.float32)).abs()
    rel = diff / (k.to(torch.float32).abs() + 1e-3)
    # FP8 e4m3fn has 3 mantissa bits ⇒ ~6% RNE step at the boundary, ~12.5%
    # at the wider end (mantissa step / value at the boundary above).  Median
    # rel error around 0.1 is the expected ceiling for random inputs.
    assert rel.median() < 0.2, f"median rel error too high: {rel.median()}"
