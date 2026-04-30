"""batched_transpose / transpose: stubbed.

The skill repo's `references/worked_examples/batched_transpose.md` contains a
~150-line annotated FlyDSL skeleton for this kernel.  The skeleton exercises
the LDS allocator, swizzle-padded LDS layout, custom thread-value layout
(decoded from the TileKernels `T.Fragment(forward_fn=...)`) and a strided
input via `buffer_ops.create_buffer_resource` + manual byte offsets.

The unverified pieces are:
- The decoded `thr_layout`/`val_layout` matching the original `loop_layout`'s
  forward_fn precisely.
- The exact spelling of `lds.as_memref(...)` for using the LDS as a
  `partition_S` source -- this method may not exist on `SmemPtr`; the fallback
  is a manual SmemPtr.load / register-stage / global write.

The kernel is sized for gfx950 (160 KB LDS, wave64).
"""

import torch
from fly_tile_kernels._stub import not_yet_ported


def transpose(x: torch.Tensor) -> torch.Tensor:
    not_yet_ported("transpose",
                   "see references/worked_examples/batched_transpose.md for the skeleton")


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    not_yet_ported("batched_transpose",
                   "see references/worked_examples/batched_transpose.md for the skeleton")
