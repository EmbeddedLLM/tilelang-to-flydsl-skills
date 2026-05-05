"""Test FlyvLLM mhc_pre against a torch reference.

The reference replays the same chain as the FlyDSL pipeline:
  1. tf32 GEMM (x.float() @ fn.T)
  2. Per-token sqrsum
  3. RMS rsqrt
  4. mixes = (x @ fn.T) * rsqrt
  5. Sigmoid splits → pre / post / comb_unnormalized
  6. Sinkhorn → comb_mix
  7. pre_apply_mix → layer_input

That matches what ``vllm.model_executor.layers.mhc.mhc_pre`` computes
under the FlyDSL backend.
"""

import pytest
import torch

from fly_vllm.mhc import mhc_pre


def _sinkhorn(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Reference sinkhorn matching the FlyDSL kernel's algorithm."""
    y = x.softmax(-1) + eps
    y = y / (y.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        y = y / (y.sum(-1, keepdim=True) + eps)
        y = y / (y.sum(-2, keepdim=True) + eps)
    return y


def mhc_pre_ref(
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
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    M3 = hc_mult * 2 + hc_mult * hc_mult
    K = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    n = residual_flat.shape[0]
    x = residual_flat.view(n, K)

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        gemm = x.float() @ fn.T              # (n, M3)
        sqrsum = x.float().square().sum(-1)  # (n,)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    rsqrt = (sqrsum / K + rms_eps).rsqrt()  # (n,)
    mixes = gemm * rsqrt.unsqueeze(-1)       # (n, M3)

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post = torch.sigmoid(
        mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    ) * hc_post_mult_value
    comb = (mixes[:, 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]).view(
        n, hc_mult, hc_mult
    )
    comb_normed = _sinkhorn(comb, sinkhorn_repeat, hc_sinkhorn_eps)

    layer_input = (residual_flat.float() * pre.unsqueeze(-1)).sum(-2).bfloat16()

    post_mix = post.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_normed.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


@pytest.mark.parametrize('n1', [512, 1024, 2048])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
@pytest.mark.parametrize('hc_mult', [4])
def test_mhc_pre(n1: int, hidden_size: int, hc_mult: int) -> None:
    torch.manual_seed(0)
    n0 = 1
    M3 = hc_mult * (2 + hc_mult)
    device = 'cuda'

    residual = (
        torch.randn((n0, n1, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((M3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((M3,), dtype=torch.float, device=device) * 0.1

    rms_eps = 1e-6
    hc_pre_eps = 1e-6
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 10

    post_fly, comb_fly, layer_fly = mhc_pre(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat,
    )
    post_ref, comb_ref, layer_ref = mhc_pre_ref(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat,
    )

    torch.testing.assert_close(post_fly, post_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb_fly, comb_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(layer_fly, layer_ref, rtol=1e-2, atol=1e-2)
