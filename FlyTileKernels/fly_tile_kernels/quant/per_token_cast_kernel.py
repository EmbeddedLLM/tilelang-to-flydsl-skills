"""per_token_cast / per_token_cast_with_sf_only / per_token_cast_with_precomputed_sf: stubbed.

Per-token (row-wise) FP8/FP4 quantisation with optional pre-quantised input
(`with_sf`).  Cast formats: e4m3 (FP8), e2m1 (FP4 packed in int8), e5m6
(special path delegated to per_token_cast_to_e5m6).

The TileKernels original is ~300 lines and uses several FlyDSL-unfriendly
features: `T.alloc_fragment` with custom forward_fn, `T.reshape` to fold the
column dimension into vector groups, `T.reduce_absmax` with a reshape, and
heavy use of `T.macro` helpers (`get_sf_and_inv`, `load_sf`, `store_sf`).

Porting plan:
1. Use `BufferCopy128b` for the input load (16 bytes / token / step).
2. Replace `T.reshape(frag, (M, K // V, V))` with a `Vector` carrier of length
   K/V whose lanes carry `V`-element sub-vectors.
3. The `reduce_absmax` becomes a per-thread `Vec.reduce(MAX)` over abs values
   followed by the wave/block reduction in `_flydsl_helpers.py`.
4. The fp8 e4m3 store needs `BufferCopy16b` per token (or a vectorised variant
   for larger blocks); the rocdl HW pack instructions on gfx950 may apply.
5. The `with_sf` (input pre-quantised) path needs an additional sf-load pass
   plus the fp8 -> f32 unscale.
"""

import torch
from typing import Optional, Union
from fly_tile_kernels._stub import not_yet_ported
from fly_tile_kernels.quant.types import QuantTensor


def per_token_cast(x, fmt: str, num_per_channels: int, x_block_size=None,
                   use_tma_aligned_col_major_sf: bool = False, round_sf: bool = False,
                   use_packed_ue8m0: bool = False) -> QuantTensor:
    not_yet_ported("per_token_cast",
                   "needs Vec.reduce(MAX) + block_reduce + sf-store skeleton")


def per_token_cast_with_sf_only(*args, **kwargs):
    not_yet_ported("per_token_cast_with_sf_only", "same as per_token_cast")


def per_token_cast_with_precomputed_sf(*args, **kwargs):
    not_yet_ported("per_token_cast_with_precomputed_sf", "same as per_token_cast")
