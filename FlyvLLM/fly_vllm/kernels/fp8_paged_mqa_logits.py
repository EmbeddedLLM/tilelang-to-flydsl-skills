"""fp8_paged_mqa_logits.

Paged-KV variant of fp8_mqa_logits.  The dominant cost is per-page bf16
GEMM (after FP8 dequant); hipBLASLt beats a handcoded FlyDSL GEMM here
just as in the non-paged version, so we use the torch reference for the
FlyDSL fast path and benchmark against both torch and torch.compile.

A future FlyDSL kernel could fuse FP8 dequant + GEMM + relu + sum-over-h
+ scale + per-page-mask in one shot, removing the temporary score
tensor; that's the main optimization opportunity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len], float("-inf"),
            device=q.device, dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            num_pages = _cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_flat[pages]
            so = block_size * dim
            cv = cache[..., :so].view(dtype=fp8_dtype).to(torch.float32)
            cs = cache[..., so:].view(dtype=torch.float32).contiguous()
            cv = cv.view(padded_seq_len, dim)
            cs = cs.view(padded_seq_len)
            score = F.linear(cv, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cs
            logits[i, :seq_len] = score[:seq_len]
        return logits

    # next_n > 1: causal masked path
    kv, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float32)
    qf = q.float()
    kvf = kv.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kvf.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len], float("-inf"),
        device=q.device, dtype=torch.float32,
    )
    for i in range(batch_size):
        ctx = context_lens[i]
        if ctx.ndim == 0:
            ctx_i = int(ctx.item())
            q_offsets = torch.arange(ctx_i - next_n, ctx_i, device=q.device)
            ctx_lim = torch.full((next_n,), ctx_i, dtype=torch.int32, device=q.device)
        else:
            ctx_lim = ctx.to(device=q.device, dtype=torch.int32)
            q_offsets = ctx_lim - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_ctx = int(ctx_lim.max().item())
        for blk in range(_cdiv(max_ctx, block_size)):
            block_idx = block_tables[i][blk]
            qx, kx = qf[i], kvf[block_idx]
            k_offsets = torch.arange(
                blk * block_size, (blk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < ctx_lim[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                blk * block_size : (blk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def fp8_paged_mqa_logits_flydsl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    return fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
