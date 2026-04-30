"""topk_gate: stubbed.

One warp per token, repeats `argmax` num_topk times to extract top-k.  Each
iteration reads the per-warp `scores_fragment`, finds the max via
`T.reduce_max`, then uses a tie-breaking `alloc_reducer('min', replication='all')`
to pick the smaller index when scores are tied.

Translates well to FlyDSL using `Vector.reduce(MAX/MIN)` + the wave_reduce
helper from `_flydsl_helpers.py` for cross-lane reduction.  But it also needs
- a way to broadcast the chosen index to lane 0 for the LDS write
- an integer BufferCopy skeleton for the int64 output

The wave-butterfly reduction helpers in `_flydsl_helpers.py` already handle
the cross-lane part; the missing piece is the int64 store.
"""

import torch
from fly_tile_kernels._stub import not_yet_ported


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    not_yet_ported(
        "topk_gate",
        "needs int64 BufferCopy store; otherwise the wave_reduce helper handles the math",
    )
