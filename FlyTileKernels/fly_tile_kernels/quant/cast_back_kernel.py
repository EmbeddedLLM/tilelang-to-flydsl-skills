"""cast_back / per_token_cast_back: stubbed."""

from fly_tile_kernels._stub import not_yet_ported


def cast_back(*args, **kwargs):
    not_yet_ported("cast_back", "fp8/fp4 dequantise back to bf16/fp32 with per-token or per-block sf")


def per_token_cast_back(*args, **kwargs):
    not_yet_ported("per_token_cast_back", "see cast_back")
