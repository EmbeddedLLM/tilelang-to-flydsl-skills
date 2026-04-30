"""MoE-side kernel-internal helpers.

The TileKernels original puts a `@T.macro` `get_topk_group_idx` here that is
called from inside `top2_sum_gate_kernel`.  In FlyDSL there is no macro
mechanism, so the equivalent code is inlined inside the calling kernel.
This module is kept (empty) so import paths line up with the upstream test
suite.
"""
