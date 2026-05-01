"""mhc_multilayer_recompute: torch-fallback (kernel port pending).

Chains ``mhc_pre_apply_mix`` and ``mhc_post`` for each layer.  The original
TileLang kernel fuses these into one persistent kernel that ping-pongs
residual fragments through registers across layers; our fallback uses the
same per-op kernels back-to-back via the modeling-layer wrappers, which
produces bit-exact (``torch.equal``) output because the wrappers' fwd
kernels are already torch-fallbacks themselves.

Math (per layer ``i``):
    layer_input[i] = mhc_pre_apply_mix(residual, pre_mix[i])
    if i < num_post:
        residual = mhc_post(layer_output[i], residual,
                             post_mix[i], comb_mix[i])
        residual_out[i] = residual
"""

import torch

from fly_tile_kernels.modeling.mhc.ops.post import mhc_post
from fly_tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix


def mhc_multilayer_recompute(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)
    assert num_layers == len(layer_input_list)
    assert num_post == len(post_mix_list) == len(comb_mix_list) == len(residual_list)
    assert num_post == num_layers - 1 or num_post == num_layers, (
        f'post count ({num_post}) must be num_layers-1 or num_layers (num_layers={num_layers})'
    )
    assert num_layers > 0

    residual = initial_residual
    for i in range(num_layers):
        mhc_pre_apply_mix(residual, pre_mix_list[i], out=layer_input_list[i])
        if i < num_post:
            mhc_post(
                layer_output_list[i],
                residual,
                post_mix_list[i],
                comb_mix_list[i],
                out=residual_list[i],
            )
            residual = residual_list[i]
