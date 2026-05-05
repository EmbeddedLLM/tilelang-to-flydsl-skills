"""FlyvLLM: FlyDSL kernels for the vLLM MHC layer (gfx950 target).

Provides drop-in replacements for the TileLang kernels in
``vllm.model_executor.layers.mhc``:

- ``mhc_pre(residual, fn, hc_scale, hc_base, ...) -> (post_mix, comb_mix, layer_input)``
- ``mhc_post(x, residual, post_layer_mix, comb_res_mix) -> out``

Inference-only — no autograd / backward path.
"""

from fly_vllm.mhc import mhc_post, mhc_pre

__all__ = ["mhc_pre", "mhc_post"]
