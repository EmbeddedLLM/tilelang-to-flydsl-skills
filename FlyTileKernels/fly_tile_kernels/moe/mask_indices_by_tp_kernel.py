"""mask_indices_by_tp: stubbed.

Per-element masking of a 2-D int64/int32 index tensor.  Pure read+compute+
write per element with no shared memory or reductions.

Blocking issue:
- FlyDSL does not expose direct `buf[idx]` indexing on a buffer-backed tensor
  for integer dtypes; the standard pattern is via `fx.slice` + `logical_divide`
  + `BufferCopy*b` atoms.  For int64 (8B) the natural copy width is 64b; for
  int32 (4B) it is 32b.  The masking math itself maps cleanly to FlyDSL once
  the load/store pattern is established.

To complete the port:
1. Wrap `indices` and `masked_indices` with `fx.rocdl.make_buffer_tensor`.
2. Treat the (num_tokens, num_topk) tensor as a 1-D run of length
   `num_total = num_tokens * num_topk`; stride to per-thread chunks via
   `logical_divide(buf, fx.make_layout(VEC, 1))`.
3. Per element: load via `fx.copy_atom_call(BufferCopy{32,64}b, ..., r_in)`,
   apply the masking math (already pure arithmetic on `fx.Int64`/`fx.Int32`),
   store via the same atom in reverse direction.
"""

import torch
from fly_tile_kernels._stub import not_yet_ported


def mask_indices_by_tp(indices: torch.Tensor, n: int, num_ep_ranks: int,
                      tp_rank: int, num_tp_ranks: int) -> torch.Tensor:
    not_yet_ported(
        "mask_indices_by_tp",
        "needs integer BufferCopy load/store skeleton for int64/int32 indices",
    )
