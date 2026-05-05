"""FlyDSL port of ``tilelang-ascend/examples/deepseek_v4/sparse_flash_attention.py``.

DeepSeek V4 sparse flash attention.  For each query (b, m) and each head h,
gather KV at the topk indices, compute scaled QK scores, mask invalid
indices to -inf, apply softmax with an additive ``attn_sink`` term in the
denominator, and produce probs @ V.

Inputs / output shape (matches the upstream Ascend example bit-for-bit):

    q          : (b, m, h, d)        bf16
    kv         : (b, n, d)           bf16
    attn_sink  : (h,)                fp32
    topk_idxs  : (b, m, topk)        int32     (-1 = invalid / pad)
    output     : (b, m, h, d)        fp32

Algorithm (FlashAttention-2 with sink, per (b, m, h) block):

    state: m = -inf, l = 0, o[d] = 0
    for tile in [0, topk, BLOCK_N):
        idx = topk_idxs[b, m, tile : tile+BLOCK_N]               (-1 → masked)
        kv_tile = kv[b, idx, :]                                  (zero-padded)
        s[n]    = sum_d q[d] * kv_tile[n, d] * scale             (-inf if idx == -1)
        m_new   = max(m, max_n s[n])
        alpha   = exp(m - m_new)
        e[n]    = exp(s[n] - m_new)
        l       = alpha * l + sum_n e[n]
        o[d]    = alpha * o[d] + sum_n e[n] * kv_tile[n, d]
        m       = m_new
    l    += exp(attn_sink[h] - m)
    out[d] = o[d] / l                                            (0 if m == -inf)

Design (gfx950, FlyDSL 0.1.2):
- Grid: (b*m, h, 1) — one block per (query, head)
- Block: 128 threads (= 2 waves on gfx950)
- Each thread holds OUT_PER_THREAD = d / NUM_THREADS output elements (4 for d=512)
- BLOCK_N = 32 KV positions per online-softmax tile
- All reductions over BLOCK_N use a 2-stage (wave + LDS) butterfly

LDS layout per block (~34 KB):
    q_smem   : (d,) fp32                — promoted bf16 query, broadcast
    k_smem   : (BLOCK_N, d) bf16        — gathered KV tile
    idx_smem : (BLOCK_N,) i32           — indices for this tile
    exp_smem : (BLOCK_N,) fp32          — exp(s - m) for this tile
    red_smem : (2,) fp32                — cross-wave reduction scratch

Critical FlyDSL idioms used here (gfx950, FlyDSL 0.1.2):
- The d-loop in the score / output dot products uses a runtime ``scf.for``
  with explicit ``init=[...]`` iter args instead of ``range_constexpr``.  At
  d=512 the unrolled IR for a 512-element dot product blows up compile
  time and register pressure on AMD; the runtime loop keeps the per-thread
  basic block bounded.
- The for-op's results aren't surfaced to user code, so the loop's final
  iter-arg is captured via a single-element register memref written from
  inside the loop body and re-loaded after the loop.
- ``Boolean.select`` returns a raw ArithValue; rewrap with ``fx.Float32(.)``
  / ``fx.Int32(.)`` to call ``.to(...)``.
- ``SmemPtr.get()`` caches its view; force one ``smem.get()`` per pointer at
  the top of the kernel so the cached value dominates later uses outside
  the if-statements they may first appear in.
- ``SmemPtr.load/store`` indices must be MLIR ``index`` Values; we coerce
  via the small ``_idx`` helper.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, gpu, math as fxmath, arith
from flydsl._mlir import ir
from flydsl._mlir.ir import InsertionPoint
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.extras import types as _T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


# ---------------------------------------------------------------------------
# Kernel parameters (gfx950).
# ---------------------------------------------------------------------------

NUM_THREADS = 128
WARP_SIZE = 64
BLOCK_N = 32


def _dyn(x):
    """Force the AST rewriter to treat the enclosing if as dynamic
    (the rewriter only converts ``if`` whose test contains a Call)."""
    return x


def _idx(v):
    """Coerce Python int / fx.Int32 / ir.Value to an MLIR ``index`` ir.Value."""
    if isinstance(v, int):
        return _arith.constant(_T.index(), v)
    if hasattr(v, "ir_value"):
        raw = v.ir_value()
    elif isinstance(v, ir.Value):
        raw = v
    else:
        raw = v
    if isinstance(raw, ir.Value) and isinstance(raw.type, ir.IndexType):
        return raw
    return _arith.IndexCastOp(_T.index(), raw).result


def _alloc_bytes(allocator: SmemAllocator, byte_count: int, align: int = 16) -> int:
    """Bump-allocator: return the byte offset for a ``byte_count``-byte slab."""
    offset = allocator._align(allocator.ptr, align)
    allocator.ptr = offset + byte_count
    return offset


# ---------------------------------------------------------------------------
# Wave / block reductions (branch-free version — no Python ``if`` in helpers,
# which the AST rewriter doesn't reach).
# ---------------------------------------------------------------------------

def _wave_reduce(value, mode: str):
    fm = arith.FastMathFlags.fast
    w = value
    log2 = int(math.log2(WARP_SIZE))
    width_v = fx.Int32(WARP_SIZE).ir_value()
    for shift in range_constexpr(log2):
        off = WARP_SIZE // (2 << shift)
        off_v = fx.Int32(off).ir_value()
        peer = w.shuffle_xor(off_v, width_v)
        if mode == "max":
            w = w.maximumf(peer)
        elif mode == "sum":
            w = w.addf(peer, fastmath=fm)
        else:
            raise ValueError(mode)
    return w


def _block_reduce(value, mode: str, red_smem, tid):
    """2-stage block reduction over NUM_THREADS=128 = 2 waves.

    Each wave's lanes hold the same wave-reduced value, so we let every
    lane write to ``red_smem[wave]`` (benign race — identical writes), then
    every thread re-reads both wave slots and reduces again.  Result is
    broadcast to every thread.
    """
    lane = tid % fx.Int32(WARP_SIZE)
    wave = tid // fx.Int32(WARP_SIZE)
    n0 = fx.Float32(float("-inf")) if mode == "max" else fx.Float32(0.0)

    w = _wave_reduce(value, mode)
    SmemPtr.store(red_smem, w, [_idx(wave)])
    gpu.barrier()

    in_range = lane < fx.Int32(2)
    lane_safe = fx.Int32(in_range.select(lane.ir_value(), fx.Int32(0).ir_value()))
    v = fx.Float32(SmemPtr.load(red_smem, [_idx(lane_safe)]))
    v = fx.Float32(in_range.select(v.ir_value(), n0.ir_value()))
    v = _wave_reduce(v, mode)
    return v


# ---------------------------------------------------------------------------
# Kernel factory.
# ---------------------------------------------------------------------------

def _build_kernel(h: int, d: int, topk: int, scale: float):
    H = h
    D = d
    TOPK = topk
    NUM_TILES = TOPK // BLOCK_N
    assert TOPK % BLOCK_N == 0, "topk must be divisible by BLOCK_N=32"
    assert D % NUM_THREADS == 0, "d must be divisible by NUM_THREADS=128"
    OUT_PER_THREAD = D // NUM_THREADS

    # ---- LDS plan ----------------------------------------------------------
    allocator = SmemAllocator(
        None, arch="gfx950",
        global_sym_name=f"smem_sparse_attn_{H}_{D}_{TOPK}",
    )
    off_q = _alloc_bytes(allocator, D * 4)                # fp32
    off_k = _alloc_bytes(allocator, BLOCK_N * D * 2)      # bf16
    off_idx = _alloc_bytes(allocator, BLOCK_N * 4)        # i32
    off_exp = _alloc_bytes(allocator, BLOCK_N * 4)        # fp32
    off_red = _alloc_bytes(allocator, 2 * 4)              # fp32

    @flyc.kernel
    def kernel(
        q_bf16:    fx.Tensor,    # (b, m, h, d) bf16    (treated as (b*m, h, d))
        kv_bf16:   fx.Tensor,    # (b, n, d) bf16        (treated as (b*n, d) by host)
        attn_sink: fx.Tensor,    # (h,) fp32
        topk_idxs: fx.Tensor,    # (b, m, topk) int32    (treated as (b*m, topk))
        out_f32:   fx.Tensor,    # (b, m, h, d) fp32     (treated as (b*m, h, d))
        bm:        fx.Int32,     # b * m  (number of (batch, query) pairs)
        n_per_batch: fx.Int32,   # n      (so we can compute batch from bm_idx)
        m_per_batch: fx.Int32,   # m      (used to recover (b, m) → batch offset)
    ):
        # block_idx.x = bm index = b_i * m + m_i
        # block_idx.y = head index (0..h-1)
        bm_idx = fx.block_idx.x
        hq = fx.block_idx.y
        tid = fx.thread_idx.x

        # batch_id = bm_idx // m
        batch_id = bm_idx // m_per_batch

        q_buf = fx.rocdl.make_buffer_tensor(q_bf16)
        kv_buf = fx.rocdl.make_buffer_tensor(kv_bf16)
        idx_buf = fx.rocdl.make_buffer_tensor(topk_idxs)
        sink_buf = fx.rocdl.make_buffer_tensor(attn_sink)
        out_buf = fx.rocdl.make_buffer_tensor(out_f32)

        # ---- LDS views -----------------------------------------------------
        base = allocator.get_base()
        q_smem = SmemPtr(base, off_q, fx.Float32.ir_type, shape=(D,))
        k_smem = SmemPtr(base, off_k, fx.BFloat16.ir_type, shape=(BLOCK_N, D))
        idx_smem = SmemPtr(base, off_idx, fx.Int32.ir_type, shape=(BLOCK_N,))
        exp_smem = SmemPtr(base, off_exp, fx.Float32.ir_type, shape=(BLOCK_N,))
        red_smem = SmemPtr(base, off_red, fx.Float32.ir_type, shape=(2,))
        # Force view construction at top scope (cached views must dominate).
        q_smem.get()
        k_smem.get()
        idx_smem.get()
        exp_smem.get()
        red_smem.get()

        # ---- Register memref scratch (1-element) for buffer_copy ----------
        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        f32_reg_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        i32_reg_ty = fx.MemRefType.get(
            fx.Int32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        atom_32_f = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        atom_32_i = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        vec_ty_1_f = ir.VectorType.get([1], fx.Float32.ir_type)

        # ---- Phase 1: load q[bm_idx, hq, :] into q_smem (fp32) ------------
        q_row = fx.slice(q_buf, (bm_idx, hq, None))
        q_row_div = fx.logical_divide(q_row, fx.make_layout(1, 1))
        for d_base in range_constexpr(0, D, NUM_THREADS):
            d_off = d_base + tid
            if _dyn(d_off < fx.Int32(D)):
                rr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(q_row_div, (None, d_off)), rr)
                bv = fx.BFloat16(
                    fxvec.extract(fx.memref_load_vec(rr), static_position=[0])
                )
                SmemPtr.store(q_smem, bv.to(fx.Float32), [_idx(d_off)])
        gpu.barrier()

        # Load attn_sink[hq] (broadcast — every thread has it).
        sink_div = fx.logical_divide(sink_buf, fx.make_layout(1, 1))
        rs = fx.memref_alloca(f32_reg_ty, s_lay)
        fx.copy_atom_call(atom_32_f, fx.slice(sink_div, (None, hq)), rs)
        sink_val = fx.Float32(
            fxvec.extract(fx.memref_load_vec(rs), static_position=[0])
        )

        # ---- Streaming softmax state (per-thread, registers) --------------
        m = fx.Float32(float("-inf"))
        l = fx.Float32(0.0)
        # OUT_PER_THREAD per-thread output accumulators held as Numeric values
        # (Python-level mutation across the range_constexpr-unrolled tile +
        # n + j loops; the unrolled tracer turns these into SSA values).
        # Each thread tid owns h_d_slot[j] = tid + j * NUM_THREADS.
        o_acc = [fx.Float32(0.0) for _ in range(OUT_PER_THREAD)]

        # ---- Phase 2: per-tile loop --------------------------------------
        for tile in range_constexpr(NUM_TILES):
            tile_start = tile * BLOCK_N

            # 2a. Load BLOCK_N indices into idx_smem.
            idx_row = fx.slice(idx_buf, (bm_idx, None))
            idx_row_div = fx.logical_divide(idx_row, fx.make_layout(1, 1))
            if _dyn(tid < fx.Int32(BLOCK_N)):
                pos = fx.Int32(tile_start) + tid
                ri = fx.memref_alloca(i32_reg_ty, s_lay)
                fx.copy_atom_call(atom_32_i, fx.slice(idx_row_div, (None, pos)), ri)
                raw = fx.Int32(
                    fxvec.extract(fx.memref_load_vec(ri), static_position=[0])
                )
                # Out-of-range or sentinel → -1 (let the gather code zero it).
                ok = (raw >= fx.Int32(0)) & (raw < n_per_batch)
                idx_v = fx.Int32(ok.select(raw.ir_value(), fx.Int32(-1).ir_value()))
                SmemPtr.store(idx_smem, idx_v, [_idx(tid)])
            gpu.barrier()

            # 2b. Cooperative gather: kv[batch_id, idx[n], d] -> k_smem[n, d].
            #     For -1 indices, write a 0 into k_smem (matches upstream
            #     gather_sparse_kv masking).
            ELEMS = BLOCK_N * D
            for off_base in range_constexpr(0, ELEMS, NUM_THREADS):
                eidx = off_base + tid
                if _dyn(eidx < fx.Int32(ELEMS)):
                    n = eidx // fx.Int32(D)
                    d_idx = eidx % fx.Int32(D)
                    idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(n)]))
                    valid = idx_n >= fx.Int32(0)
                    safe_idx = fx.Int32(valid.select(
                        idx_n.ir_value(), fx.Int32(0).ir_value(),
                    ))
                    # kv is laid out as (b, n, d) but we treat it as (b*n, d)
                    # in the runner (it's contiguous), so the linear row id
                    # for batch b_i, position p is b_i * n + p.
                    global_kv_row = batch_id * n_per_batch + safe_idx
                    rb = fx.memref_alloca(bf16_reg_ty, s_lay)
                    kv_row = fx.slice(kv_buf, (global_kv_row, None))
                    kv_row_div = fx.logical_divide(kv_row, fx.make_layout(1, 1))
                    fx.copy_atom_call(atom_16, fx.slice(kv_row_div, (None, d_idx)), rb)
                    bv = fx.BFloat16(
                        fxvec.extract(fx.memref_load_vec(rb), static_position=[0])
                    )
                    # Zero out invalid: store 0 instead of the loaded value.
                    bv_safe = fx.BFloat16(valid.select(
                        bv.ir_value(), fx.BFloat16(0.0).ir_value(),
                    ))
                    SmemPtr.store(k_smem, bv_safe, [_idx(n), _idx(d_idx)])
            gpu.barrier()

            # 2c. Score compute: every thread participates.  The first
            #     BLOCK_N threads each compute one score; threads beyond
            #     BLOCK_N produce -inf.  Use scf.for over d to keep the
            #     basic block bounded (d=512 → unrolled = 512 FMAs which
            #     blows up at compile time on AMD).
            tid_in_blockn = tid < fx.Int32(BLOCK_N)
            tid_safe = fx.Int32(tid_in_blockn.select(
                tid.ir_value(), fx.Int32(0).ir_value(),
            ))
            idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(tid_safe)]))
            valid = (idx_n >= fx.Int32(0)) & tid_in_blockn

            # Runtime d-loop with one fp32 iter-arg.  Capture the final
            # value in a register memref (FlyDSL doesn't expose for-op
            # results to user code).
            acc_cap = fx.memref_alloca(f32_reg_ty, s_lay)
            fx.memref_store_vec(
                fxvec.from_elements(vec_ty_1_f, [fx.Float32(0.0)]), acc_cap,
            )
            for d_iv, [acc] in range(0, D, 1, init=[fx.Float32(0.0).ir_value()]):
                d_int = fx.Int32(d_iv)
                qv = fx.Float32(SmemPtr.load(q_smem, [_idx(d_int)]))
                kvv = fx.BFloat16(
                    SmemPtr.load(k_smem, [_idx(tid_safe), _idx(d_int)])
                ).to(fx.Float32)
                new_acc = fx.Float32(acc) + qv * kvv
                fx.memref_store_vec(
                    fxvec.from_elements(vec_ty_1_f, [new_acc]), acc_cap,
                )
                yield new_acc.ir_value()
            acc = fx.Float32(
                fxvec.extract(fx.memref_load_vec(acc_cap), static_position=[0])
            )
            acc = acc * fx.Float32(scale)
            score = fx.Float32(valid.select(
                acc.ir_value(),
                fx.Float32(float("-inf")).ir_value(),
            ))

            # 2d. Block-reduce max(score) across all 128 threads.
            tile_m = _block_reduce(score, "max", red_smem, tid)
            m_new = m.maximumf(tile_m)

            # 2e. exp(score - m_new), with -inf-safety on m_new.
            alpha = fx.Float32(fxmath.exp(m - m_new))
            m_was_minf = m == fx.Float32(float("-inf"))
            alpha = fx.Float32(m_was_minf.select(
                fx.Float32(1.0).ir_value(), alpha.ir_value(),
            ))

            e = fx.Float32(fxmath.exp(score - m_new))
            tile_lonely = m_new == fx.Float32(float("-inf"))
            e = fx.Float32(tile_lonely.select(
                fx.Float32(0.0).ir_value(), e.ir_value(),
            ))
            if _dyn(tid_in_blockn):
                SmemPtr.store(exp_smem, e, [_idx(tid)])
            gpu.barrier()

            # 2f. Block-reduce sum(exp) across all 128 threads.
            my_e = fx.Float32(tid_in_blockn.select(
                e.ir_value(), fx.Float32(0.0).ir_value(),
            ))
            tile_l = _block_reduce(my_e, "sum", red_smem, tid)

            # 2g. Update l (scalar register).
            l = alpha * l + tile_l

            # 2h. Output update — register-resident.  First rescale by alpha,
            #     then accumulate sum_n e_n * v_n for each of OUT_PER_THREAD
            #     outputs.  Both n and j loops are range_constexpr-unrolled
            #     (BLOCK_N=32, OUT_PER_THREAD≤4 → ≤128 unrolled FMAs per
            #     thread per tile, fine for the compiler).
            for j in range_constexpr(OUT_PER_THREAD):
                o_acc[j] = alpha * o_acc[j]
            for n in range_constexpr(BLOCK_N):
                e_n = fx.Float32(SmemPtr.load(exp_smem, [_idx(n)]))
                for j in range_constexpr(OUT_PER_THREAD):
                    h_d = tid + fx.Int32(j * NUM_THREADS)
                    v_n = fx.BFloat16(
                        SmemPtr.load(k_smem, [_idx(n), _idx(h_d)])
                    ).to(fx.Float32)
                    o_acc[j] = o_acc[j] + e_n * v_n

            m = m_new
            gpu.barrier()

        # ---- Finalize ------------------------------------------------------
        # l += exp(sink - m), with -inf-safety.
        sink_term = fx.Float32(fxmath.exp(sink_val - m))
        m_was_minf = m == fx.Float32(float("-inf"))
        sink_term = fx.Float32(m_was_minf.select(
            fx.Float32(0.0).ir_value(), sink_term.ir_value(),
        ))
        l = l + sink_term

        lonely = m == fx.Float32(float("-inf"))

        # Write OUT_PER_THREAD outputs per thread at strided positions.
        out_row = fx.slice(out_buf, (bm_idx, hq, None))
        out_row_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
        for j in range_constexpr(OUT_PER_THREAD):
            h_d = tid + fx.Int32(j * NUM_THREADS)
            out_v = fx.Float32(lonely.select(
                fx.Float32(0.0).ir_value(),
                (o_acc[j] / l).ir_value(),
            ))
            outr = fx.memref_alloca(f32_reg_ty, s_lay)
            fx.memref_store_vec(
                fxvec.from_elements(vec_ty_1_f, [out_v]), outr,
            )
            fx.copy_atom_call(atom_32_f, outr, fx.slice(out_row_div, (None, h_d)))

    @flyc.jit
    def launch(
        q_bf16:    fx.Tensor,
        kv_bf16:   fx.Tensor,
        attn_sink: fx.Tensor,
        topk_idxs: fx.Tensor,
        out_f32:   fx.Tensor,
        bm:        fx.Int32,
        n_per_batch: fx.Int32,
        m_per_batch: fx.Int32,
        stream:    fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        # SmemAllocator.finalized is sticky across compiles; reset so the
        # global memref is re-emitted whenever cache=0 forces recompile.
        allocator.finalized = False
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(
            q_bf16, kv_bf16, attn_sink, topk_idxs, out_f32,
            bm, n_per_batch, m_per_batch,
        ).launch(grid=(bm, H, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    def runner(q, kv, attn_sink, topk_idxs, out, bm, n, m):
        if bm == 0:
            return
        launch(
            q.detach(), kv.detach(), attn_sink.detach(), topk_idxs.detach(),
            out.detach(), int(bm), int(n), int(m),
        )

    return runner


@lru_cache(maxsize=64)
def _get_kernel(h: int, d: int, topk: int, scale: float):
    return _build_kernel(h, d, topk, float(scale))


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """FlashAttention with sparse-gathered KV and an attn_sink term in the
    softmax denominator.

    Args:
        q:          (b, m, h, d) bf16 — queries
        kv:         (b, n, d) bf16    — KV (gathered by topk_idxs)
        attn_sink:  (h,) fp32         — per-head sink term in the softmax denom
        topk_idxs:  (b, m, topk) int32 — KV positions to attend to;
                                          -1 = invalid / padded
        scale:      softmax pre-scale (typically 1/sqrt(d))

    Returns:
        (b, m, h, d) fp32 — attention output
    """
    assert q.dtype == torch.bfloat16, "q must be bf16"
    assert kv.dtype == torch.bfloat16, "kv must be bf16"
    assert attn_sink.dtype == torch.float32, "attn_sink must be fp32"
    assert topk_idxs.dtype == torch.int32, "topk_idxs must be int32"
    b, m, h, d = q.shape
    b2, n, d2 = kv.shape
    assert b2 == b and d2 == d, "kv shape mismatch"
    assert attn_sink.shape == (h,), f"attn_sink shape {attn_sink.shape} != ({h},)"
    bb, mm, topk = topk_idxs.shape
    assert bb == b and mm == m, "topk_idxs shape mismatch"
    assert topk % BLOCK_N == 0, f"topk ({topk}) must be divisible by BLOCK_N={BLOCK_N}"
    assert d % NUM_THREADS == 0, f"d ({d}) must be divisible by NUM_THREADS={NUM_THREADS}"

    out = torch.empty(b, m, h, d, dtype=torch.float32, device=q.device)
    # Flatten (b, m) → bm, and (b, n) → b*n in kv (still contiguous).
    q_flat = q.contiguous().view(b * m, h, d)
    kv_flat = kv.contiguous().view(b * n, d)
    idx_flat = topk_idxs.contiguous().view(b * m, topk)
    out_flat = out.view(b * m, h, d)

    runner = _get_kernel(h, d, topk, scale)
    runner(q_flat, kv_flat, attn_sink, idx_flat, out_flat, b * m, n, m)
    return out
