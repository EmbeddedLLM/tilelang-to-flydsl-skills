"""per_block_cast / *_with_precomputed_sf / *_with_sf_only: stubbed."""

from fly_tile_kernels._stub import not_yet_ported


def per_block_cast(*args, **kwargs):
    not_yet_ported("per_block_cast", "two-axis (block) reduction + fp8 store")


def per_block_cast_with_precomputed_sf(*args, **kwargs):
    not_yet_ported("per_block_cast_with_precomputed_sf", "same as per_block_cast")


def per_block_cast_with_sf_only(*args, **kwargs):
    not_yet_ported("per_block_cast_with_sf_only", "same as per_block_cast")
