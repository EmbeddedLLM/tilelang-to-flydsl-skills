"""engram_gate_fwd / engram_gate_bwd: stubbed."""

from fly_tile_kernels._stub import not_yet_ported


def engram_gate_fwd(*args, **kwargs):
    not_yet_ported("engram_gate_fwd", "engram gate forward (fused with rmsnorm)")


def engram_gate_bwd(*args, **kwargs):
    not_yet_ported("engram_gate_bwd", "engram gate backward (fused with rmsnorm grad)")
