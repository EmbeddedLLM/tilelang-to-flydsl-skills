"""top2_sum_gate: stubbed.

This is the most complex MoE kernel (~425 lines of TileLang).  It does:
- Per-token softmax / sigmoid / sqrtsoftplus / identity scoring
- Optional bias addition
- Top-k group selection (warp-level top-2 sum per group)
- Stable top-k expert selection via repeated argmax
- Weight normalisation
- Logical -> physical expert mapping with prime-number hashing
- ETP rank masking

Porting plan when unblocked:
1. Use one warp per token (32 threads on RDNA / 64 on CDNA -- gfx950 is wave64
   so adjust `warp_size` accordingly).  The TileKernels original assumes
   warp_size=32; gfx950 uses 64.  Either rewrite for wave64 or run with one
   wave per two tokens.
2. Replace `T.alloc_reducer('min', replication='all')` with the wave_reduce
   helper from _flydsl_helpers.py.
3. Replace `T.shfl_xor` / `T.shfl_sync` with `value.shuffle_xor(off, WARP_SIZE)`.
4. The per-group top-2 sum (`get_topk_group_idx` macro) becomes an inline
   helper; the warp-level argmax pattern is the same as topk_gate.
5. The final integer (int64) writes need the int64 BufferCopy skeleton.

Wave-size mismatch is a non-trivial design issue: the TileKernels original
encodes `warp_size = 32` deeply (lane indexing, shuffle masks, the
`num_routed_experts <= warp_size` constraint).  A clean port to gfx950
would either:
(a) Run two tokens per wave64, partitioning lanes 0..31 vs 32..63;
(b) Generalise to wave64, doubling the per-thread work and the shuffle masks.

Option (b) is simpler but slower for small num_routed_experts.
"""

import torch
from typing import Optional
from fly_tile_kernels._stub import not_yet_ported


def top2_sum_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_topk: int,
    num_topk_groups: int,
    num_groups: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
    scoring_func: str,
    mask: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    not_yet_ported(
        "top2_sum_gate",
        "complex (~425 lines) + warp_size mismatch (TileKernels original assumes wave32; "
        "gfx950 is wave64). See module docstring for porting plan.",
    )
