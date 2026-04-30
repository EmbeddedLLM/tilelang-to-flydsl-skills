"""Internal FlyDSL helpers shared by kernels in this package.

These are *thin* wrappers that hide the FlyDSL boiler-plate: dtype dispatch,
copy-atom width selection, the wave/block reduction butterfly, etc.  They
are kept private (leading underscore) because none of this surface is part
of the package's public API.

Targeted at gfx950 (CDNA 4 / MI350 / MI355X):
- wave size 64
- 160 KB LDS / CU
- OCP fp8 variants (Float8E4M3FN, Float8E5M2) -- not the FNUZ variants
- bf16 has hardware pack instructions (USE_HW_CVT_PK_BF16_F32)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, const_expr, arith
from flydsl.expr.vector import ReductionOp


# ---------------------------------------------------------------------------
# Architecture constants for gfx950.  We hard-pin to gfx950 per the user's
# instruction; kernels that need to be portable can read get_rocm_arch().
# ---------------------------------------------------------------------------

GFX950_WARP_SIZE = 64
GFX950_LDS_BYTES = 160 * 1024


# ---------------------------------------------------------------------------
# Dtype dispatch.  Maps a torch.dtype (or T.dtype string when called from a
# context that has translated it to str) to the FlyDSL Numeric class.
# ---------------------------------------------------------------------------

def torch_dtype_to_fx(dtype: torch.dtype):
    """Map a torch.dtype to the FlyDSL Numeric class for gfx950.

    NB: gfx950 implements the OCP fp8 variants (E4M3FN / E5M2), *not* the
    FNUZ variants used by gfx94x.  Picking the wrong variant produces silently
    wrong outputs.
    """
    if dtype == torch.float32:
        return fx.Float32
    if dtype == torch.float16:
        return fx.Float16
    if dtype == torch.bfloat16:
        return fx.BFloat16
    if dtype == torch.float8_e4m3fn:
        return fx.Float8E4M3FN
    if dtype == torch.float8_e5m2:
        return fx.Float8E5M2
    if dtype == torch.int8:
        return fx.Int8
    if dtype == torch.int16:
        return fx.Int16
    if dtype == torch.int32:
        return fx.Int32
    if dtype == torch.int64:
        return fx.Int64
    if dtype == torch.uint8:
        return fx.Uint8
    if dtype == torch.uint16:
        return fx.Uint16
    if dtype == torch.uint32:
        return fx.Uint32
    if dtype == torch.uint64:
        return fx.Uint64
    if dtype == torch.bool:
        return fx.Boolean
    raise ValueError(f"unsupported torch dtype for gfx950 FlyDSL: {dtype}")


def torch_dtype_str(dtype: torch.dtype) -> str:
    """Compact string identifier used as a Constexpr cache key."""
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# Copy-atom width selection.  Given a register memref of `(VEC, 1)` for a
# given dtype, pick the largest BufferCopy*b that holds it in one access.
# Falls back to a smaller atom when alignment isn't satisfied.
# ---------------------------------------------------------------------------

def pick_buffer_copy_atom(elem_dt, vec_width: int):
    """Return a (atom_factory, n_atom_calls) pair sized to vec_width * elem_bits.

    `atom_factory` is the unbound `fx.rocdl.BufferCopyNb` constructor; the
    caller MUST invoke it (e.g. ``atom_factory()``) inside an MLIR context
    (i.e. inside the `@flyc.kernel` body).  Constructing the atom outside a
    context raises a context-missing error.

    Caller invokes the atom `n_atom_calls` times to cover the full vector.
    The atom always emits a single 32/64/128-bit AMD `buffer_load`/store.
    """
    bits = vec_width * elem_dt.width
    if bits == 16:
        return fx.rocdl.BufferCopy16b, 1
    if bits == 32:
        return fx.rocdl.BufferCopy32b, 1
    if bits == 64:
        return fx.rocdl.BufferCopy64b, 1
    if bits == 128:
        return fx.rocdl.BufferCopy128b, 1
    if bits == 256:
        # Two 128-bit copies.
        return fx.rocdl.BufferCopy128b, 2
    if bits == 512:
        return fx.rocdl.BufferCopy128b, 4
    raise ValueError(
        f"vec_width * bits = {bits} has no matching BufferCopy atom; "
        f"reduce vec_width or change dtype"
    )


def decompose_buffer_copy(elem_dt, vec_width: int):
    """Decompose a vector of `vec_width` elements of `elem_dt` into a sequence
    of (atom_factory, elem_count) copy chunks that together cover the vector.

    Returns a list of (atom_factory, elem_count) pairs.  The factories are
    unbound `fx.rocdl.BufferCopyNb` constructors; call them inside the
    `@flyc.kernel` body.  Each chunk advances the offset by ``elem_count``
    elements; the caller is responsible for issuing one ``copy_atom_call``
    per chunk against the appropriate slice.

    Strategy: greedy descending widths in {128b, 64b, 32b}.  For element
    widths that don't divide 32b evenly the function falls through to 16b
    chunks; on widths smaller than 16b an exception is raised.
    """
    elem_bits = elem_dt.width
    total_bits = vec_width * elem_bits
    chunks: list = []
    remaining = total_bits
    while remaining > 0:
        for width_bits, factory in (
            (128, fx.rocdl.BufferCopy128b),
            (64, fx.rocdl.BufferCopy64b),
            (32, fx.rocdl.BufferCopy32b),
            (16, fx.rocdl.BufferCopy16b),
        ):
            if remaining >= width_bits and (width_bits % elem_bits) == 0:
                count = width_bits // elem_bits
                chunks.append((factory, count))
                remaining -= width_bits
                break
        else:
            raise ValueError(
                f"cannot decompose {total_bits} bits with element width "
                f"{elem_bits}; smallest atom is 16b"
            )
    return chunks


def make_register_memref(elem_dt, length: int):
    """Allocate a register memref of `(length, 1)`.

    Returns (memref_type, layout); call `fx.memref_alloca(*pair)` to get the
    actual `Vector`-compatible storage.
    """
    mref_ty = fx.MemRefType.get(
        elem_dt.ir_type,
        fx.LayoutType.get(length, 1),
        fx.AddressSpace.Register,
    )
    layout = fx.make_layout(length, 1)
    return mref_ty, layout


# ---------------------------------------------------------------------------
# Wave / block reductions.  See references/idioms.md §4 for derivation.
# Operate on `Numeric` values (the result of a Vector.reduce or a per-thread
# scalar).  Mode is one of "max", "sum", "min", "and", "or".
# ---------------------------------------------------------------------------

def wave_reduce(value, mode: str, *, warp_size: int = GFX950_WARP_SIZE):
    """Butterfly wave reduction.  Result lives in every lane."""
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
        elif const_expr(mode == "and"):
            w = w & peer
        elif const_expr(mode == "or"):
            w = w | peer
        else:
            raise ValueError(f"unknown reduce mode {mode}")
    return w


def neutral(mode: str, dtype):
    """The identity element for a reduction `mode` and value `dtype`."""
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
    if mode == "and":
        return dtype(-1)
    if mode == "or":
        return dtype(0)
    raise ValueError(f"unknown reduce mode {mode}")


def block_reduce(value, mode: str, s_red, tid, *,
                 num_threads: int,
                 warp_size: int = GFX950_WARP_SIZE,
                 dtype=fx.Float32):
    """Two-stage workgroup reduction.

    `s_red` must be an `SmemPtr` of length `(num_threads + warp_size - 1) //
    warp_size` typed to `dtype`.  Returns the reduced value broadcast to
    every lane.
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
