"""reduce_fused: stubbed."""

import torch
from fly_tile_kernels._stub import not_yet_ported


def reduce_fused(*args, **kwargs):
    not_yet_ported(
        "reduce_fused",
        "weighted token reduction using token->topk fan-in tables; needs gather + accum loop",
    )
