"""group_count: stubbed.

Original semantics: persistent grid of `num_sms*2` blocks; each block strides
over the token range, atomically increments a per-block LDS i32 table, then
atomically adds the LDS table into the global i32 output.

Blocking issue (gfx950 / FlyDSL):
- Need integer LDS atomics (`ds_atomic_add_i32`) and integer global atomics
  (`global_atomic_add_i32` or buffer-atomic equivalent).  FlyDSL currently
  exposes `raw_ptr_buffer_atomic_fadd` / `_fmax` (float-only) in
  `flydsl/expr/rocdl/__init__.py`; the integer counterparts need to be added
  before this kernel can be ported.

When that surface lands, the port follows the standard pattern:
- `SmemAllocator.allocate_array(flyT.i32, align(num_groups, NUM_THREADS))`
- LDS clear loop using `range_constexpr(0, LDS_LEN, NUM_THREADS)` + SmemPtr.store
- Outer persistent loop using `range(gtid, num_tokens, GLOBAL_STRIDE)`
- Per-row inner unroll over `num_topk` with the integer-LDS atomic
- Barrier
- LDS-flush loop with the integer-global atomic
"""

import torch
from fly_tile_kernels._stub import not_yet_ported


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    not_yet_ported("group_count", "integer LDS/global atomics not exposed in FlyDSL")
