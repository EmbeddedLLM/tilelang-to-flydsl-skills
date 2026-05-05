"""Test FlyvLLM mhc_post against the torch reference."""

import pytest
import torch

from fly_vllm.mhc import mhc_post


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation matching the math in vLLM mhc.py."""
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [4096])
@pytest.mark.parametrize('h', [1280, 2560, 7168])
def test_mhc_post(n0: int, n1: int, h: int) -> None:
    torch.manual_seed(0)
    mhc_mult = 4
    device = 'cuda'

    x = torch.randn(n0, n1, h, dtype=torch.bfloat16, device=device)
    residual = torch.randn(n0, n1, mhc_mult, h, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn(n0, n1, mhc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(n0, n1, mhc_mult, mhc_mult, dtype=torch.float32, device=device)

    out_fly = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    out_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out_fly, out_ref)
