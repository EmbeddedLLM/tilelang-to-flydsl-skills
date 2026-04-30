"""expand_to_fused / expand_to_fused_with_sf: stubbed."""

import torch
from fly_tile_kernels._stub import not_yet_ported


def expand_to_fused(*args, **kwargs):
    not_yet_ported("expand_to_fused", "indirect-gather with per-expert offset table")


def expand_to_fused_with_sf(*args, **kwargs):
    not_yet_ported("expand_to_fused_with_sf", "indirect-gather with per-token sf factors")
