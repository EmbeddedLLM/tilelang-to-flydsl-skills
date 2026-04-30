"""topk_sum_and_topk_group_idx: stubbed."""

import torch
from fly_tile_kernels._stub import not_yet_ported


def topk_sum_and_topk_group_idx(*args, **kwargs):
    not_yet_ported(
        "topk_sum_and_topk_group_idx",
        "shares warp-size and atomic blockers with top2_sum_gate",
    )
