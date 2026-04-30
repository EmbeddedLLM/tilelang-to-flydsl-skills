"""inplace_unique_group_indices: stubbed.

Per-token bit-set deduplication of a (num_tokens, num_topk) int64 group-index
tensor.  Each token's threads carry a 128-bit bitmap (two u64) in registers;
the kth load is masked out if its bit was already set, otherwise the bit is
recorded and the load is kept.

Blocking issue:
- `T.alloc_local((2,), T.uint64)` becomes `fx.memref_alloca` of a u64 register
  memref of length 2 -- straightforward.
- Reading `group_indices[i, j]` and *writing back* `-1` to the same slot when
  duplicate is found needs an i64 buffer load/store skeleton (same as
  mask_indices_by_tp).
- The bit-twiddling math (`<< (group_idx % 64)`) maps cleanly once the
  load/store skeleton is in place.

When the int64 BufferCopy skeleton is ready, the kernel itself is ~50 lines.
"""

import torch
from fly_tile_kernels._stub import not_yet_ported


def inplace_unique_group_indices(group_indices: torch.Tensor, num_groups: int) -> None:
    not_yet_ported(
        "inplace_unique_group_indices",
        "needs integer BufferCopy skeleton (same as mask_indices_by_tp)",
    )
