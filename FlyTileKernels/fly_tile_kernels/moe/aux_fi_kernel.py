"""aux_fi: stubbed. Same atomic-count pattern as group_count, plus an f32 scaling on flush."""

import torch
from fly_tile_kernels._stub import not_yet_ported


def aux_fi(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    not_yet_ported(
        "aux_fi",
        "same blockers as group_count (integer LDS atomic) + need f32 atomic_add_global on the flush",
    )
