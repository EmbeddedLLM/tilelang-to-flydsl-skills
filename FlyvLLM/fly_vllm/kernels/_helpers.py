"""Shared FlyDSL helpers — wave/block reductions, copy-atom helpers.

Targeted at gfx950 (wave size 64).
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, const_expr, arith


WARP_SIZE = 64


def wave_reduce(value, mode: str, *, warp_size: int = WARP_SIZE):
    """Butterfly wave reduction over a single wave.  Result lives in every lane."""
    fm_fast = arith.FastMathFlags.fast
    w = value
    log2 = int(math.log2(warp_size))
    for shift_exp in range_constexpr(log2):
        off = warp_size // (2 << shift_exp)
        peer = w.shuffle_xor(off, warp_size)
        if const_expr(mode == "max"):
            w = w.maximumf(peer)
        elif const_expr(mode == "min"):
            w = w.minimumf(peer)
        elif const_expr(mode == "sum"):
            w = w.addf(peer, fastmath=fm_fast)
        elif const_expr(mode == "max_int"):
            w = w.maximumi(peer)
        elif const_expr(mode == "min_int"):
            w = w.minimumi(peer)
        elif const_expr(mode == "sum_int"):
            w = w + peer
        else:
            raise ValueError(f"unknown reduce mode {mode}")
    return w


def neutral(mode: str, dtype):
    if mode == "max":
        return dtype(float("-inf"))
    if mode == "min":
        return dtype(float("inf"))
    if mode == "sum":
        return dtype(0.0)
    if mode == "max_int":
        return dtype(-(1 << 62))
    if mode == "min_int":
        return dtype((1 << 62) - 1)
    if mode == "sum_int":
        return dtype(0)
    raise ValueError(f"unknown reduce mode {mode}")


def block_reduce(value, mode: str, s_red, tid, *,
                 num_threads: int,
                 warp_size: int = WARP_SIZE,
                 dtype=fx.Float32):
    """Two-stage workgroup reduction.

    ``s_red`` must be an ``SmemPtr`` of length
    ``ceil(num_threads / warp_size)`` typed to ``dtype``.  Returns the
    reduced value broadcast to every lane.
    """
    from flydsl.utils.smem_allocator import SmemPtr

    red_slots = max(1, (num_threads + warp_size - 1) // warp_size)
    if const_expr(red_slots == 1):
        return wave_reduce(value, mode, warp_size=warp_size)

    lane = tid % warp_size
    wave = tid // warp_size
    n0 = neutral(mode, dtype)

    w = wave_reduce(value, mode, warp_size=warp_size)
    if lane == 0:
        SmemPtr.store(s_red, w, [wave])
    gpu.barrier()

    if wave == 0:
        in_range = lane < red_slots
        lane_safe = in_range.select(lane, 0)
        v = SmemPtr.load(s_red, [lane_safe])
        v = in_range.select(v, n0)
        v = wave_reduce(v, mode, warp_size=warp_size)
        if lane == 0:
            SmemPtr.store(s_red, v, [0])
    gpu.barrier()
    return SmemPtr.load(s_red, [0])
