"""FlyvLLM MHC layer: drop-in replacement for
``vllm.model_executor.layers.mhc`` (TileLang ⇒ FlyDSL on gfx950).

Public API:
- ``mhc_pre(residual, fn, hc_scale, hc_base, ...) -> (post_mix, comb_mix, layer_input)``
- ``mhc_post(x, residual, post_layer_mix, comb_res_mix) -> out``

Both functions are inference-only (no autograd) and match the tensor
shapes/dtypes of the upstream vLLM implementation.

Internal pipeline of ``mhc_pre`` (decomposed from the deeply-fused
TileLang kernel):

    1. ``tf32_hc_prenorm_gemm`` → split-K matmul + per-row sqrsum
       (torch fallback wrapping ``torch.matmul`` with allow_tf32=True;
       hipBLASLt covers this for the supported sizes).
    2. ``mhc_norm_split_kernel`` → reduce splits, RMS rsqrt, sigmoid
       split into pre / post / comb_unnormalized.
    3. ``sinkhorn_kernel`` → comb_unnormalized → comb_mix.
    4. ``pre_apply_mix_kernel`` → residual * pre_mix → layer_input.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from fly_vllm.kernels.mhc_norm_split_kernel import get_norm_split_kernel
from fly_vllm.kernels.mhc_post_kernel import get_mhc_post_kernel
from fly_vllm.kernels.pre_apply_mix_kernel import get_pre_apply_mix_kernel
from fly_vllm.kernels.sinkhorn_kernel import get_sinkhorn_kernel
from fly_vllm.kernels.tf32_hc_prenorm_gemm import tf32_hc_prenorm_gemm


@lru_cache(maxsize=64)
def _compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    """Mirror of ``vllm.model_executor.layers.mhc.compute_num_split``.

    Returns the chosen split-K factor.  On ROCm, since
    ``tf32_hc_prenorm_gemm`` is a torch fallback that collapses the split
    back into a full matmul, the choice doesn't affect correctness — but
    we keep the same logic so the public n_splits arg has the same
    meaning across the two backends.
    """
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_k = max(1, n_sms // max(1, grid_size))
    if k is not None:
        num_block_k = (k + block_k - 1) // block_k
        split_k = min(split_k, max(1, num_block_k // 4))
    return max(split_k, 1)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for mHC pre block.

    Args:
        residual:   (..., hc_mult, hidden_size)            bf16
        fn:         (hc_mult3, hc_mult * hidden_size)      fp32
        hc_scale:   (3,)                                   fp32
        hc_base:    (hc_mult3,)                            fp32
        rms_eps:           RMS normalisation epsilon
        hc_pre_eps:        pre-mix epsilon
        hc_sinkhorn_eps:   sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat:   number of sinkhorn iterations
        n_splits:          split-K factor (informational on ROCm)

    Returns:
        post_mix:    (..., hc_mult, 1)        fp32
        comb_mix:    (..., hc_mult, hc_mult)  fp32
        layer_input: (..., hidden_size)       bf16
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape == (hc_mult3, hc_hidden_size), (fn.shape, hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]

    block_k = 64
    block_m = 64
    grid_size = (num_tokens + block_m - 1) // block_m
    n_splits = _compute_num_split(block_k, hc_hidden_size, grid_size)

    # ---- 1. tf32 GEMM + per-token sqrsum (split-K).
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3,
        dtype=torch.float32, device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens,
        dtype=torch.float32, device=residual.device,
    )
    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    # ---- 2. Reduce splits + RMS rsqrt + sigmoid split.
    pre_mix = torch.empty(
        num_tokens, hc_mult,
        dtype=torch.float32, device=residual.device,
    )
    post_mix_flat = torch.empty(
        num_tokens, hc_mult,
        dtype=torch.float32, device=residual.device,
    )
    comb_unnormed = torch.empty(
        num_tokens, hc_mult, hc_mult,
        dtype=torch.float32, device=residual.device,
    )
    if num_tokens > 0:
        ns_kernel = get_norm_split_kernel(
            hc_mult, hidden_size, n_splits,
            rms_eps, hc_pre_eps, hc_post_mult_value,
        )
        ns_kernel(
            gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
            pre_mix, post_mix_flat, comb_unnormed, num_tokens,
        )

    # ---- 3. Sinkhorn-normalise comb_unnormed → comb_mix (in-place same shape).
    comb_mix_3d = torch.empty_like(comb_unnormed)
    if num_tokens > 0:
        sinkhorn_kernel = get_sinkhorn_kernel(hc_mult, sinkhorn_repeat, hc_sinkhorn_eps)
        sinkhorn_kernel(comb_unnormed, comb_mix_3d, num_tokens)

    # ---- 4. pre_apply_mix.
    layer_input = torch.empty(
        num_tokens, hidden_size,
        dtype=torch.bfloat16, device=residual.device,
    )
    if num_tokens > 0:
        pa_kernel = get_pre_apply_mix_kernel(hc_mult, hidden_size)
        pa_kernel(residual_flat, pre_mix, layer_input, num_tokens)

    # Restore outer shape and singleton dim on post_mix.
    post_mix = post_mix_flat.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix_3d.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Forward pass for mHC post block.

    Args:
        x:              (..., hidden_size)             bf16
        residual:       (..., hc_mult, hidden_size)    bf16
        post_layer_mix: (..., hc_mult, 1)              fp32
        comb_res_mix:   (..., hc_mult, hc_mult)        fp32

    Returns:
        out: (..., hc_mult, hidden_size) bf16
    """
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = int(torch.tensor(outer_shape).prod().item()) if outer_shape else 1

    residual_flat = residual.reshape(num_tokens, hc_mult, hidden_size).contiguous()
    x_flat = x.reshape(num_tokens, hidden_size).contiguous()
    post_flat = post_layer_mix.reshape(num_tokens, hc_mult, 1).squeeze(-1).contiguous()
    comb_flat = comb_res_mix.reshape(num_tokens, hc_mult, hc_mult).contiguous()

    out_flat = torch.empty_like(residual_flat)
    if num_tokens > 0:
        runner = get_mhc_post_kernel(hc_mult, hidden_size)
        runner(comb_flat, residual_flat, post_flat, x_flat, out_flat, num_tokens)
    return out_flat.view(*outer_shape, hc_mult, hidden_size)
