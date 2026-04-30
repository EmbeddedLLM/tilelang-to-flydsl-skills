"""per_channel_cast_fused: stubbed.

The most complex quant kernel — fuses an optional token-expansion gather
(`pos_to_token`) with per-channel scaling, optional input rescale (with_sf),
and a per-channel sf write-out.  See tile_kernels/quant/per_channel_cast_fused_kernel.py
for the original ~135-line TileLang source; the FlyDSL port mirrors the
3-pass structure (load+amax, sf compute, normalise+store) using the wave/block
reduction helpers in _flydsl_helpers.py.
"""

from fly_tile_kernels._stub import not_yet_ported


def per_channel_cast_fused(*args, **kwargs):
    not_yet_ported("per_channel_cast_fused", "fused gather + col-reduce + scale + store")
