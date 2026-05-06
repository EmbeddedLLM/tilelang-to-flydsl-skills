"""sparse_attn_prefill: FlashAttention-style sparse attention prefill.

For each query position (s_q, h_q) the kernel
- gathers KV at the topk indices,
- masks invalid (idx == -1 or topk_length-padded) positions to -inf,
- runs an online softmax (FlashAttention-2 style) that fuses the
  ``q @ K``, ``softmax``, and ``probs @ V`` steps with no intermediate
  ``score`` tensor,
- folds the optional ``attn_sink`` into the softmax denominator,
- handles "lonely" queries (all indices invalid) by writing zero output.

Math (per (sq, hq), all summations over the topk axis):
    s[n]   = scale * sum_d q[sq, hq, d] * kv[idx[n], d]      (-inf if invalid)
    m      = max_n s[n]
    l      = sum_n exp(s[n] - m)
    l'     = l + exp(attn_sink[hq] - m)                       (if present)
    out[h] = sum_n exp(s[n] - m) * kv[idx[n], h]  / l'        (h < head_dim)
    out    = 0  if m == -inf

Layout:
- Grid:  (s_q, h_q)
- Block: 128 threads (= 2 waves on gfx950)
- BLOCK_N = 64 KV positions per online-softmax tile
- All reductions over BLOCK_N use a 2-stage (wave + LDS) butterfly.

LDS layout per block (≈25 KB on gfx950):
    q_smem    : (d_qk,) fp32        — promoted bf16 query, broadcast
    k_smem    : (BLOCK_N, d_qk) bf16 — gathered KV tile
    idx_smem  : (BLOCK_N,) i32       — indices for this tile (-1 == invalid)
    exp_smem  : (BLOCK_N,) fp32      — exp(s - m) for this tile
    red_smem  : (2,) fp32            — cross-wave reduction scratch

Currently specialised to ``d_qk = 192``, ``head_dim = 128``, ``topk % BLOCK_N == 0``.
General configurations fall through to the torch reference.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, gpu, arith, rocdl
from flydsl._mlir import ir
from flydsl._mlir.ir import InsertionPoint
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.extras import types as _T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import (
    SmemAllocator, SmemPtr, get_mlir_type_size,
)


# Fold into scale so the softmax inner can use bare v_exp_f32 (rocdl.exp2)
# without paying for a per-call log2e multiply on (m_old - m_new) and
# (score - m_new). All of m / score / alpha / e are kept in the log2
# domain throughout the kernel; outputs are unaffected.
LOG2E = 1.4426950408889634


def _fast_exp2(x: fx.Float32) -> fx.Float32:
    """Bare v_exp_f32 (no range reduction).

    Safe wherever the argument is <= 0 (so the implicit -inf path returns 0
    and there's no overflow concern). For the lonely-tile case (m_new == -inf
    so arg == NaN) callers already select 0 via tile_lonely / m_was_minf.
    """
    return fx.Float32(rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _idx(v):
    """Coerce Python int / fx.Int32 / ir.Value to an MLIR index ir.Value."""
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


NUM_THREADS = 128
WARP_SIZE = 64


def _block_n_for(d_qk: int) -> int:
    """Pick BLOCK_N to fit the gfx9 64KB per-block LDS budget.

    LDS K-tile slab = BLOCK_N * d_qk * 2 bytes (bf16).
    - d_qk=192 → BLOCK_N=64 → 24 KB (fast V3 path).
    - d_qk=576 → BLOCK_N=64 would be 73 KB (overflows); BLOCK_N=32 → 36 KB.
    """
    if d_qk <= 256:
        return 64
    return 32


def _dyn(x):
    """Force the AST rewriter to treat the enclosing if as dynamic."""
    return x


def _alloc_bytes(allocator: SmemAllocator, byte_count: int, align: int = 16) -> int:
    """Bump-allocator: returns the byte offset where a ``byte_count``-byte
    slab lives.  Called at kernel-build time *before* any MLIR context
    exists, so we side-step the ir_type accessors and pass byte counts
    directly."""
    offset = allocator._align(allocator.ptr, align)
    allocator.ptr = offset + byte_count
    return offset


def _wave_reduce(value, mode: str):
    """Butterfly reduction within one wave (size 64).  mode in {max, sum}."""
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


def _group_reduce_sum(value, group_size: int):
    """Sum-reduce across ``group_size`` consecutive lanes within a wave.

    For ``group_size=4``, executes butterfly shuffles with masks 1 and 2;
    after the reduction every lane in the group of 4 holds the group sum.
    Operates within the ambient wave of 64 (groups never cross wave
    boundaries since BLOCK_N divides 64 in our configurations).
    """
    if group_size <= 1:
        return value
    fm = arith.FastMathFlags.fast
    w = value
    log2 = int(math.log2(group_size))
    assert (1 << log2) == group_size, "group_size must be a power of 2"
    width_v = fx.Int32(WARP_SIZE).ir_value()
    for shift in range_constexpr(log2):
        off_v = fx.Int32(1 << shift).ir_value()
        peer = w.shuffle_xor(off_v, width_v)
        w = w.addf(peer, fastmath=fm)
    return w


def _block_reduce(value, mode: str, red_smem, tid):
    """Two-stage block reduction (NUM_THREADS = 128, two waves).

    Branch-free: every lane writes the same wave-reduced value into the
    shared slot for its wave (benign race — all writes are identical), then
    every thread re-reads the two slots and reduces again.  Avoids the
    Python ``if lane == 0`` that the FlyDSL AST rewriter doesn't reach
    inside a helper function body.
    """
    lane = tid % fx.Int32(WARP_SIZE)
    wave = tid // fx.Int32(WARP_SIZE)
    n0 = fx.Float32(float("-inf")) if mode == "max" else fx.Float32(0.0)

    w = _wave_reduce(value, mode)
    # All lanes of a wave have the same `w` after wave_reduce, so they all
    # write the same value to red_smem[wave] — no functional race.
    SmemPtr.store(red_smem, w, [_idx(wave)])
    gpu.barrier()

    in_range = lane < fx.Int32(2)
    lane_safe = fx.Int32(in_range.select(lane.ir_value(), fx.Int32(0).ir_value()))
    v = fx.Float32(SmemPtr.load(red_smem, [_idx(lane_safe)]))
    v = fx.Float32(in_range.select(v.ir_value(), n0.ir_value()))
    v = _wave_reduce(v, mode)
    return v


def _build_kernel(
    h_q: int, d_qk: int, head_dim: int, topk: int,
    use_attn_sink: bool, use_topk_length: bool,
    scale: float,
):
    HQ = h_q
    DK = d_qk
    HD = head_dim
    TOPK = topk
    # Bake LOG2E into the scale so the score multiply produces values
    # already in the log2 domain (allowing rocdl.exp2 to replace exp).
    SCALE_LOG2E = scale * LOG2E
    BLOCK_N = _block_n_for(DK)
    NUM_TILES = TOPK // BLOCK_N
    # For DK > 256 (DSv4 sizes), the d-dim score loop must be a runtime
    # ``scf.for`` rather than a constexpr unroll — a 576-element unrolled
    # dot product blows up compile time + register pressure on AMD.  V3
    # (DK=192) keeps the unrolled fast path.
    USE_RUNTIME_D_LOOP = DK > 256
    assert TOPK % BLOCK_N == 0, "topk must be divisible by BLOCK_N"
    assert HD % NUM_THREADS == 0, (
        f"head_dim ({HD}) must be a multiple of NUM_THREADS ({NUM_THREADS})"
    )
    OUT_PER_THREAD = HD // NUM_THREADS  # registers held per thread for o

    # Threads-per-score: split the dot product across THREADS_PER_SCORE
    # threads per score; each thread sums DK / THREADS_PER_SCORE muladds, then
    # we do a partial-wave reduction across the group.  For NUM_THREADS=128
    # and BLOCK_N=64 (V3) this is 2; for BLOCK_N=32 (V4) this is 4.  Without
    # this, 75% of threads do nothing useful in the score compute on V4.
    THREADS_PER_SCORE = NUM_THREADS // BLOCK_N
    assert NUM_THREADS % BLOCK_N == 0
    assert DK % THREADS_PER_SCORE == 0, (
        f"d_qk ({DK}) must be divisible by NUM_THREADS/BLOCK_N "
        f"({THREADS_PER_SCORE})"
    )
    DK_PER_LANE = DK // THREADS_PER_SCORE

    # K_SMEM_PAD: pad each k_smem row with 2 bf16 elements when the natural
    # row stride (DK bf16 = DK/2 4-byte words) is a multiple of 32 banks, so
    # score-compute reads where threads each load a different row at the same
    # column don't all hit the same bank.
    # Empirically: the V3 path (DK<=256, unrolled inner loops) gets slower
    # with padding (LLVM seems to schedule the non-power-of-2 stride worse),
    # so we only apply the pad on the V4 (runtime-loop) path.
    K_SMEM_PAD = (
        2 if USE_RUNTIME_D_LOOP and (DK * 2 // 4) % 32 == 0 else 0
    )
    DK_STRIDE = DK + K_SMEM_PAD

    # ---- LDS plan ----------------------------------------------------------
    allocator = SmemAllocator(
        None, arch="gfx950",
        global_sym_name=f"smem_sparse_prefill_{HQ}_{DK}_{HD}_{TOPK}",
    )
    off_q = _alloc_bytes(allocator, DK * 4)                   # fp32
    off_k = _alloc_bytes(allocator, BLOCK_N * DK_STRIDE * 2)  # bf16, padded
    off_idx = _alloc_bytes(allocator, BLOCK_N * 4)            # i32
    off_exp = _alloc_bytes(allocator, BLOCK_N * 4)            # fp32
    off_red = _alloc_bytes(allocator, 2 * 4)                  # fp32

    @flyc.kernel
    def kernel(
        q_bf16:        fx.Tensor,    # (s_q, h_q, d_qk) bf16
        kv_bf16:       fx.Tensor,    # (s_kv, d_qk)    bf16
        indices:       fx.Tensor,    # (s_q, topk)     int32
        topk_length:   fx.Tensor,    # (s_q,)          int32 (always present, ignored if !use_topk_length)
        attn_sink:     fx.Tensor,    # (h_q,)          fp32   (always present, ignored if !use_attn_sink)
        out_bf16:      fx.Tensor,    # (s_q, h_q, head_dim) bf16
        s_kv:          fx.Int32,
    ):
        sq = fx.block_idx.x
        hq = fx.block_idx.y
        tid = fx.thread_idx.x

        q_buf = fx.rocdl.make_buffer_tensor(q_bf16)
        kv_buf = fx.rocdl.make_buffer_tensor(kv_bf16)
        idx_buf = fx.rocdl.make_buffer_tensor(indices)
        len_buf = fx.rocdl.make_buffer_tensor(topk_length)
        sink_buf = fx.rocdl.make_buffer_tensor(attn_sink)
        out_buf = fx.rocdl.make_buffer_tensor(out_bf16)

        # ---- LDS views -----------------------------------------------------
        base = allocator.get_base()
        q_smem = SmemPtr(base, off_q, fx.Float32.ir_type, shape=(DK,))
        k_smem = SmemPtr(base, off_k, fx.BFloat16.ir_type, shape=(BLOCK_N, DK_STRIDE))
        idx_smem = SmemPtr(base, off_idx, fx.Int32.ir_type, shape=(BLOCK_N,))
        exp_smem = SmemPtr(base, off_exp, fx.Float32.ir_type, shape=(BLOCK_N,))
        red_smem = SmemPtr(base, off_red, fx.Float32.ir_type, shape=(2,))
        # Force view construction in the top-level region — otherwise the
        # cached view from inside an scf.if won't dominate later uses.
        q_smem.get()
        k_smem.get()
        idx_smem.get()
        exp_smem.get()
        red_smem.get()

        # Register memref scratch (1-element) for buffer_copy.
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
        # 128-bit BufferCopy on bf16 loads 8 elements per call.  Used by the
        # V4 gather to amortize HBM latency across 8 elements per buffer-load
        # instruction (the gather is HBM-latency-bound at V4 sizes since
        # each load forces an s_waitcnt before its result can be stored
        # to LDS).
        atom_128_bf = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        bf16_reg_8_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(8, 1), fx.AddressSpace.Register,
        )
        s_lay_8 = fx.make_layout(8, 1)
        vec_ty_1_bf = ir.VectorType.get([1], fx.BFloat16.ir_type)
        vec_ty_1_f = ir.VectorType.get([1], fx.Float32.ir_type)
        # gfx9 vectorized buffer copy: 8-element groups must be aligned to 8
        # elements within a row.  All our DK values (192, 576) are multiples
        # of 8, and BLOCK_N (32, 64) ensures BLOCK_N*DK is a multiple of
        # NUM_THREADS*8 = 1024 for both V3 and V4 configs we care about.
        VEC_LOAD = 8

        # ---- Phase 1: load q[sq, hq, :] into q_smem (fp32) ----------------
        q_row = fx.slice(q_buf, (sq, hq, None))
        q_row_div = fx.logical_divide(q_row, fx.make_layout(1, 1))
        for d_base in range_constexpr(0, DK, NUM_THREADS):
            d = d_base + tid
            if _dyn(d < fx.Int32(DK)):
                rr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(atom_16, fx.slice(q_row_div, (None, d)), rr)
                bv = fx.BFloat16(
                    fxvec.extract(fx.memref_load_vec(rr), static_position=[0])
                )
                SmemPtr.store(q_smem, bv.to(fx.Float32), [_idx(d)])
        gpu.barrier()

        # Optional topk_length cap (clamp to [0, TOPK]).
        if use_topk_length:
            len_div = fx.logical_divide(len_buf, fx.make_layout(1, 1))
            ri = fx.memref_alloca(i32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_i, fx.slice(len_div, (None, sq)), ri)
            tk_len = fx.Int32(
                fxvec.extract(fx.memref_load_vec(ri), static_position=[0])
            )
        else:
            tk_len = fx.Int32(TOPK)

        sink_val = fx.Float32(float("-inf"))
        if use_attn_sink:
            sink_div = fx.logical_divide(sink_buf, fx.make_layout(1, 1))
            rs = fx.memref_alloca(f32_reg_ty, s_lay)
            fx.copy_atom_call(atom_32_f, fx.slice(sink_div, (None, hq)), rs)
            sink_val = fx.Float32(
                fxvec.extract(fx.memref_load_vec(rs), static_position=[0])
            )
            # Pre-scale to log2 domain so the finalize sink_term can use
            # rocdl.exp2(sink_val - m) (m is already log2-domain).
            sink_val = sink_val * fx.Float32(LOG2E)

        # ---- Streaming softmax state (per-thread, register) ---------------
        # Each thread owns OUT_PER_THREAD output elements at strided indices
        # h_d = tid + j*NUM_THREADS for j in [0, OUT_PER_THREAD).
        m = fx.Float32(float("-inf"))
        l = fx.Float32(0.0)
        o_regs = [fx.Float32(0.0) for _ in range(OUT_PER_THREAD)]

        # ---- Phase 2: per-tile loop ---------------------------------------
        # V4 path uses a runtime scf.for tile loop to bound the unrolled basic
        # block to one tile body's worth of code (otherwise NUM_TILES=32 ×
        # ~272 unrolled ops per tile ≈ 8.7K-op basic block which LLVM cannot
        # compile within reasonable time).  V3 stays unrolled (NUM_TILES≤4,
        # cheap to unroll, and the constant tile_start lets LLVM fold
        # addressing).
        if USE_RUNTIME_D_LOOP:
            # ---- V4 path: runtime tile loop with iter args ---------------
            # Use the `(yield ...)`-as-expression pattern to capture the
            # for-op's final results directly, instead of writing through
            # register memrefs every iteration.  Lets LLVM keep all 6 iter
            # args (m, l, o_regs[0..3]) live in VGPRs through the loop.
            tile_init = [m.ir_value(), l.ir_value()] + [
                o_regs[j].ir_value() for j in range(OUT_PER_THREAD)
            ]
            final_state = None

            for tile_iv, iter_args in range(0, NUM_TILES, 1, init=tile_init):
                m = fx.Float32(iter_args[0])
                l = fx.Float32(iter_args[1])
                o_regs = [fx.Float32(iter_args[2 + j]) for j in range(OUT_PER_THREAD)]
                tile_start_i32 = fx.Int32(tile_iv) * fx.Int32(BLOCK_N)

                # 2a. Load BLOCK_N indices into idx_smem.
                idx_row = fx.slice(idx_buf, (sq, None))
                idx_row_div = fx.logical_divide(idx_row, fx.make_layout(1, 1))
                if _dyn(tid < fx.Int32(BLOCK_N)):
                    pos = tile_start_i32 + tid
                    in_len = pos < tk_len
                    ri = fx.memref_alloca(i32_reg_ty, s_lay)
                    fx.copy_atom_call(atom_32_i, fx.slice(idx_row_div, (None, pos)), ri)
                    raw = fx.Int32(
                        fxvec.extract(fx.memref_load_vec(ri), static_position=[0])
                    )
                    ok = (raw >= fx.Int32(0)) & (raw < s_kv) & in_len
                    idx_v = fx.Int32(ok.select(raw.ir_value(), fx.Int32(-1).ir_value()))
                    SmemPtr.store(idx_smem, idx_v, [_idx(tid)])
                gpu.barrier()

                # 2b. Cooperative gather: kv[idx_smem[n], d] -> k_smem[n, d].
                #     128-bit vectorized buffer-copy (8 bf16 per load), and
                #     constexpr-unrolled so LLVM can interleave the buffer
                #     loads across passes (hiding HBM latency).  At V4 sizes
                #     (DK=576, BLOCK_N=32) this is 18 passes/tile, so per
                #     tile body 18 unrolled loads × 8 LDS stores.
                ELEMS = BLOCK_N * DK
                NUM_GATHER_PASSES = ELEMS // (NUM_THREADS * VEC_LOAD)
                assert (ELEMS % (NUM_THREADS * VEC_LOAD)) == 0, (
                    f"BLOCK_N*DK ({ELEMS}) must be a multiple of "
                    f"NUM_THREADS*VEC_LOAD ({NUM_THREADS * VEC_LOAD}) for "
                    f"the 128-bit vectorized gather"
                )
                for pass_iv in range_constexpr(NUM_GATHER_PASSES):
                    vec_idx = fx.Int32(pass_iv * NUM_THREADS) + tid
                    eidx = vec_idx * fx.Int32(VEC_LOAD)
                    n = eidx // fx.Int32(DK)
                    d = eidx % fx.Int32(DK)
                    idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(n)]))
                    valid = idx_n >= fx.Int32(0)
                    safe_idx = fx.Int32(valid.select(idx_n.ir_value(),
                                                    fx.Int32(0).ir_value()))
                    rb = fx.memref_alloca(bf16_reg_8_ty, s_lay_8)
                    kv_row = fx.slice(kv_buf, (safe_idx, None))
                    kv_row_div_8 = fx.logical_divide(kv_row, fx.make_layout(VEC_LOAD, 1))
                    vec_d = d // fx.Int32(VEC_LOAD)
                    fx.copy_atom_call(
                        atom_128_bf, fx.slice(kv_row_div_8, (None, vec_d)), rb,
                    )
                    v8 = fx.memref_load_vec(rb)
                    for k in range_constexpr(VEC_LOAD):
                        bv = fx.BFloat16(fxvec.extract(v8, static_position=[k]))
                        SmemPtr.store(k_smem, bv, [_idx(n), _idx(d + fx.Int32(k))])
                gpu.barrier()

                # 2c. Score compute with 4-thread-per-score collaboration.
                #     Each (score_idx, d_offset) lane sums DK/4=144 muladds,
                #     then a 4-way wave-shuffle reduction merges into the
                #     final score; every lane in the group ends up with the
                #     same score.
                score_idx = tid // fx.Int32(THREADS_PER_SCORE)
                d_offset = tid % fx.Int32(THREADS_PER_SCORE)
                d_base = d_offset * fx.Int32(DK_PER_LANE)
                idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(score_idx)]))
                valid = idx_n >= fx.Int32(0)

                # Use (yield expr)-as-expression to capture the for-op's
                # final iter-arg result directly — avoids the per-iteration
                # register-memref store and lets LLVM keep `acc` in a
                # register through the loop.
                final_d_acc = None
                for d_iv, [acc] in range(0, DK_PER_LANE, 1,
                                         init=[fx.Float32(0.0).ir_value()]):
                    d_int = fx.Int32(d_iv)
                    d = d_base + d_int
                    qv = fx.Float32(SmemPtr.load(q_smem, [_idx(d)]))
                    kvv = fx.BFloat16(
                        SmemPtr.load(k_smem, [_idx(score_idx), _idx(d)])
                    ).to(fx.Float32)
                    new_acc = fx.Float32(acc) + qv * kvv
                    final_d_acc = (yield new_acc.ir_value())
                acc = fx.Float32(final_d_acc)
                acc = _group_reduce_sum(acc, THREADS_PER_SCORE)
                acc = acc * fx.Float32(SCALE_LOG2E)
                score = fx.Float32(valid.select(
                    acc.ir_value(),
                    fx.Float32(float("-inf")).ir_value(),
                ))

                # 2d. Block-reduce max(score).  Group dups (THREADS_PER_SCORE
                # threads share the same score) don't affect the max.
                tile_m = _block_reduce(score, "max", red_smem, tid)
                m_new = m.maximumf(tile_m)

                # 2e. exp(score - m_new), with -inf safety.  All values are
                # already in the log2 domain (scale folds in LOG2E).
                alpha = _fast_exp2(m - m_new)
                m_was_minf = m == fx.Float32(float("-inf"))
                alpha = fx.Float32(m_was_minf.select(
                    fx.Float32(1.0).ir_value(), alpha.ir_value(),
                ))
                e = _fast_exp2(score - m_new)
                tile_lonely = m_new == fx.Float32(float("-inf"))
                e = fx.Float32(tile_lonely.select(
                    fx.Float32(0.0).ir_value(), e.ir_value(),
                ))
                # Only the first lane in each group writes to exp_smem so we
                # don't get THREADS_PER_SCORE-way write collisions.
                is_first_lane = d_offset == fx.Int32(0)
                if _dyn(is_first_lane):
                    SmemPtr.store(exp_smem, e, [_idx(score_idx)])
                gpu.barrier()

                # 2f. Block-reduce sum(exp).  Mask to one thread per score so
                # the group dups don't get summed THREADS_PER_SCORE-times.
                my_e = fx.Float32(is_first_lane.select(
                    e.ir_value(), fx.Float32(0.0).ir_value(),
                ))
                tile_l = _block_reduce(my_e, "sum", red_smem, tid)

                # 2g. l-update.
                new_l = alpha * l + tile_l

                # 2h. Output update — runtime n-loop carrying o_regs as
                #     iter args.  At BLOCK_N=32 × OUT_PER_THREAD=4 = 128
                #     unrolled FMAs × NUM_TILES=32 tiles, the constexpr
                #     version emits 4096 unrolled FMA basic-block ops, which
                #     LLVM compiles but runs poorly (likely register spill
                #     and scheduling churn).
                rescaled_o = [alpha * o_regs[j] for j in range_constexpr(OUT_PER_THREAD)]
                n_init = [rescaled_o[j].ir_value() for j in range(OUT_PER_THREAD)]
                final_n_state = None
                for n_iv, n_iter_args in range(0, BLOCK_N, 1, init=n_init):
                    n_int = fx.Int32(n_iv)
                    o_in = [fx.Float32(n_iter_args[j]) for j in range(OUT_PER_THREAD)]
                    e_n = fx.Float32(SmemPtr.load(exp_smem, [_idx(n_int)]))
                    o_out = []
                    for j in range_constexpr(OUT_PER_THREAD):
                        h_d = tid + fx.Int32(j * NUM_THREADS)
                        v_n = fx.BFloat16(
                            SmemPtr.load(k_smem, [_idx(n_int), _idx(h_d)])
                        ).to(fx.Float32)
                        o_out.append(o_in[j] + e_n * v_n)
                    yield_n_vals = [o_out[j].ir_value() for j in range(OUT_PER_THREAD)]
                    final_n_state = (yield yield_n_vals)
                new_o = [fx.Float32(final_n_state[j]) for j in range(OUT_PER_THREAD)]
                gpu.barrier()

                # Yield the new state for the next iteration AND capture the
                # for-op's final results in `final_state` (assignment-from-
                # yield pattern).
                yield_vals = [m_new.ir_value(), new_l.ir_value()] + [
                    new_o[j].ir_value() for j in range(OUT_PER_THREAD)
                ]
                final_state = (yield yield_vals)

            # Read back final state from the for-op's results.
            m = fx.Float32(final_state[0])
            l = fx.Float32(final_state[1])
            o_regs = [
                fx.Float32(final_state[2 + j]) for j in range(OUT_PER_THREAD)
            ]
        else:
            # ---- V3 path: unrolled tile loop (NUM_TILES is small) --------
            for tile in range_constexpr(NUM_TILES):
                tile_start = tile * BLOCK_N

                # 2a. Load BLOCK_N indices into idx_smem.
                idx_row = fx.slice(idx_buf, (sq, None))
                idx_row_div = fx.logical_divide(idx_row, fx.make_layout(1, 1))
                if _dyn(tid < fx.Int32(BLOCK_N)):
                    pos = fx.Int32(tile_start) + tid
                    in_len = pos < tk_len
                    ri = fx.memref_alloca(i32_reg_ty, s_lay)
                    fx.copy_atom_call(atom_32_i, fx.slice(idx_row_div, (None, pos)), ri)
                    raw = fx.Int32(
                        fxvec.extract(fx.memref_load_vec(ri), static_position=[0])
                    )
                    ok = (raw >= fx.Int32(0)) & (raw < s_kv) & in_len
                    idx_v = fx.Int32(ok.select(raw.ir_value(), fx.Int32(-1).ir_value()))
                    SmemPtr.store(idx_smem, idx_v, [_idx(tid)])
                gpu.barrier()

                # 2b. Cooperative gather (128-bit vectorized, same as V4).
                ELEMS = BLOCK_N * DK
                NUM_GATHER_PASSES_V3 = ELEMS // (NUM_THREADS * VEC_LOAD)
                assert (ELEMS % (NUM_THREADS * VEC_LOAD)) == 0, (
                    f"BLOCK_N*DK ({ELEMS}) must be a multiple of "
                    f"NUM_THREADS*VEC_LOAD ({NUM_THREADS * VEC_LOAD}) for "
                    f"the V3 vectorized gather"
                )
                for pass_iv in range_constexpr(NUM_GATHER_PASSES_V3):
                    vec_idx = fx.Int32(pass_iv * NUM_THREADS) + tid
                    eidx = vec_idx * fx.Int32(VEC_LOAD)
                    n = eidx // fx.Int32(DK)
                    d = eidx % fx.Int32(DK)
                    idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(n)]))
                    valid = idx_n >= fx.Int32(0)
                    safe_idx = fx.Int32(valid.select(idx_n.ir_value(),
                                                    fx.Int32(0).ir_value()))
                    rb = fx.memref_alloca(bf16_reg_8_ty, s_lay_8)
                    kv_row = fx.slice(kv_buf, (safe_idx, None))
                    kv_row_div_8 = fx.logical_divide(kv_row, fx.make_layout(VEC_LOAD, 1))
                    vec_d = d // fx.Int32(VEC_LOAD)
                    fx.copy_atom_call(
                        atom_128_bf, fx.slice(kv_row_div_8, (None, vec_d)), rb,
                    )
                    v8 = fx.memref_load_vec(rb)
                    for k in range_constexpr(VEC_LOAD):
                        bv = fx.BFloat16(fxvec.extract(v8, static_position=[k]))
                        SmemPtr.store(k_smem, bv, [_idx(n), _idx(d + fx.Int32(k))])
                gpu.barrier()

                # 2c. Score compute (unrolled d-loop for small DK≤256).
                tid_in_blockn = tid < fx.Int32(BLOCK_N)
                tid_safe = fx.Int32(tid_in_blockn.select(
                    tid.ir_value(), fx.Int32(0).ir_value(),
                ))
                idx_n = fx.Int32(SmemPtr.load(idx_smem, [_idx(tid_safe)]))
                valid = (idx_n >= fx.Int32(0)) & tid_in_blockn
                acc = fx.Float32(0.0)
                for d in range_constexpr(DK):
                    qv = fx.Float32(SmemPtr.load(q_smem, [_idx(d)]))
                    kvv = fx.BFloat16(
                        SmemPtr.load(k_smem, [_idx(tid_safe), _idx(d)])
                    ).to(fx.Float32)
                    acc = acc + qv * kvv
                acc = acc * fx.Float32(SCALE_LOG2E)
                score = fx.Float32(valid.select(
                    acc.ir_value(),
                    fx.Float32(float("-inf")).ir_value(),
                ))

                # 2d. Block-reduce max(score).
                tile_m = _block_reduce(score, "max", red_smem, tid)
                m_new = m.maximumf(tile_m)

                # 2e. exp(score - m_new), with -inf safety.  log2-domain
                # softmax; rocdl.exp2 emits a single v_exp_f32.
                alpha = _fast_exp2(m - m_new)
                m_was_minf = m == fx.Float32(float("-inf"))
                alpha = fx.Float32(m_was_minf.select(
                    fx.Float32(1.0).ir_value(), alpha.ir_value(),
                ))
                e = _fast_exp2(score - m_new)
                tile_lonely = m_new == fx.Float32(float("-inf"))
                e = fx.Float32(tile_lonely.select(
                    fx.Float32(0.0).ir_value(), e.ir_value(),
                ))
                if _dyn(tid_in_blockn):
                    SmemPtr.store(exp_smem, e, [_idx(tid)])
                gpu.barrier()

                # 2f. Block-reduce sum(exp).
                my_e = fx.Float32(tid_in_blockn.select(
                    e.ir_value(), fx.Float32(0.0).ir_value(),
                ))
                tile_l = _block_reduce(my_e, "sum", red_smem, tid)

                # 2g. l-update.
                l = alpha * l + tile_l

                # 2h. Output update.
                new_o = [alpha * o_regs[j] for j in range(OUT_PER_THREAD)]
                for n in range_constexpr(BLOCK_N):
                    e_n = fx.Float32(SmemPtr.load(exp_smem, [_idx(n)]))
                    for j in range_constexpr(OUT_PER_THREAD):
                        h_d = tid + fx.Int32(j * NUM_THREADS)
                        v_n = fx.BFloat16(
                            SmemPtr.load(k_smem, [_idx(n), _idx(h_d)])
                        ).to(fx.Float32)
                        new_o[j] = new_o[j] + e_n * v_n
                for j in range_constexpr(OUT_PER_THREAD):
                    o_regs[j] = new_o[j]
                m = m_new
                gpu.barrier()

        # ---- Finalize ------------------------------------------------------
        if use_attn_sink:
            # l += exp(sink - m_orig).  sink_val and m are both pre-scaled
            # by LOG2E, so exp(sink - m_orig) == exp2(sink_val - m).
            sink_term = _fast_exp2(sink_val - m)
            m_was_minf = m == fx.Float32(float("-inf"))
            # if m == -inf, we'd have sink_term = exp2(+inf) = inf; but
            # since we'll output 0 in that case anyway, just avoid NaN by
            # using 0.
            sink_term = fx.Float32(m_was_minf.select(
                fx.Float32(0.0).ir_value(), sink_term.ir_value(),
            ))
            l = l + sink_term

        lonely = m == fx.Float32(float("-inf"))

        # Write OUT_PER_THREAD outputs per thread at strided positions.
        out_row = fx.slice(out_buf, (sq, hq, None))
        out_row_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
        for j in range_constexpr(OUT_PER_THREAD):
            h_d = tid + fx.Int32(j * NUM_THREADS)
            out_v = fx.Float32(lonely.select(
                fx.Float32(0.0).ir_value(),
                (o_regs[j] / l).ir_value(),
            ))
            out_bf = out_v.to(fx.BFloat16)
            outr = fx.memref_alloca(bf16_reg_ty, s_lay)
            fx.memref_store_vec(fxvec.from_elements(vec_ty_1_bf, [out_bf]), outr)
            fx.copy_atom_call(atom_16, outr, fx.slice(out_row_div, (None, h_d)))

    @flyc.jit
    def launch(
        q_bf16:        fx.Tensor,
        kv_bf16:       fx.Tensor,
        indices:       fx.Tensor,
        topk_length:   fx.Tensor,
        attn_sink:     fx.Tensor,
        out_bf16:      fx.Tensor,
        s_kv:          fx.Int32,
        s_q:           fx.Int32,
        stream:        fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        # The SmemAllocator caches its finalized state on the instance,
        # but with FLYDSL_RUNTIME_ENABLE_CACHE=0 each launch builds a fresh
        # MLIR module, so the global memref must be re-emitted every time.
        allocator.finalized = False
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(
            q_bf16, kv_bf16, indices, topk_length, attn_sink, out_bf16, s_kv,
        ).launch(grid=(s_q, HQ, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    def runner(q, kv, indices, topk_length, attn_sink, out, s_kv, s_q):
        if s_q == 0:
            return
        launch(
            q.detach(), kv.detach(), indices.detach(),
            topk_length.detach(), attn_sink.detach(),
            out.detach(), int(s_kv), int(s_q),
        )

    return runner


@lru_cache(maxsize=64)
def _get_kernel(h_q, d_qk, head_dim, topk, use_sink, use_len, scale):
    return _build_kernel(h_q, d_qk, head_dim, topk, use_sink, use_len, scale)


# ---------------------------------------------------------------------------
# Reference path (lifted from upstream).
# ---------------------------------------------------------------------------

def rocm_ref_sparse_attn_prefill_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """If ``output`` is given (the vLLM call style — see
    ``vllm.model_executor.layers.deepseek_v4_attention`` line 1026), the
    fp32 result is cast and copied into it in place and the function
    returns ``None``.  Otherwise a fresh bf16 tensor is allocated and
    returned (used by the test harness)."""
    indices = indices.clone().squeeze(1)
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    # Match the vLLM call shape: kv may be (s_kv, 1, d_qk).
    if kv.dim() == 3:
        kv = kv.reshape(kv.shape[0], kv.shape[2])
    s_kv = kv.shape[0]
    if topk_length is not None:
        mask = torch.arange(topk, device=indices.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid = (indices < 0) | (indices >= s_kv)
    indices[invalid] = 0
    qf = q.float()
    gathered = kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
    scores = qf @ gathered.transpose(1, 2)
    scores *= scale
    scores[invalid.unsqueeze(1).expand_as(scores)] = float("-inf")
    orig_lse = torch.logsumexp(scores, dim=-1)
    lse_for_o = orig_lse
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack(
                [orig_lse, attn_sink[:h_q].view(1, h_q).expand_as(orig_lse)], dim=0,
            ), dim=0,
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
    out = probs @ gathered[..., :head_dim]
    lonely = orig_lse == float("-inf")
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    if output is None:
        return out.to(torch.bfloat16)
    output.copy_(out)  # casts fp32 -> output.dtype in-place
    return None


# ---------------------------------------------------------------------------
# FlyDSL fast path.
# ---------------------------------------------------------------------------

def _can_use_flydsl(q, kv, indices, topk, head_dim) -> bool:
    """Conditions under which we run the FlyDSL FA2-style kernel.

    The kernel handles two regimes:
    - V3 (d_qk ≤ 256, head_dim ≤ 256): score d-loop is range_constexpr-unrolled.
    - V4 (d_qk > 256): score d-loop becomes a runtime scf.for to keep the
      per-thread basic block bounded and BLOCK_N drops to 32 to fit the
      gfx9 64KB per-block LDS budget at d_qk=576.
    """
    s_q, h_q, d_qk = q.shape
    block_n = _block_n_for(d_qk)
    return (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and (head_dim % NUM_THREADS) == 0
        and (topk % block_n) == 0
        and indices.dim() == 3 and indices.shape[1] == 1
        # Absolute size cap on d_qk: at DSv4 sizes (576) the kernel still
        # works, but beyond that we'd need a different LDS plan.
        and d_qk <= 1024
        and head_dim <= 1024
    )


def rocm_ref_sparse_attn_prefill_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """vLLM-compatible entry point.

    When called from
    ``vllm.model_executor.layers.deepseek_v4_attention`` (line 1026),
    ``output`` is a pre-allocated bf16 ``(s_q, h_q, head_dim)`` slice; the
    kernel writes into it directly and returns ``None`` — no extra
    allocation or final copy.  When ``output`` is ``None`` (test harness
    path), a fresh bf16 tensor is allocated and returned.
    """
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    # vLLM passes kv as ``kv.view(-1, 1, d_qk)`` (3-D, one "kv head").  The
    # FlyDSL kernel is written for a 2-D ``(s_kv, d_qk)`` KV tensor, so
    # collapse the spurious head axis here.  No-op for 2-D kv (test path).
    if kv.dim() == 3:
        assert kv.shape[1] == 1, (
            f"sparse_attn_prefill expects exactly one kv head, got {kv.shape}"
        )
        kv = kv.reshape(kv.shape[0], kv.shape[2])
    if not _can_use_flydsl(q, kv, indices, topk, head_dim):
        return rocm_ref_sparse_attn_prefill_torch(
            q, kv, indices, topk_length, scale, head_dim, attn_sink,
            output=output,
        )
    s_kv = kv.shape[0]
    indices2 = indices.squeeze(1).contiguous().to(torch.int32)
    if topk_length is None:
        tk_len = torch.zeros(s_q, dtype=torch.int32, device=q.device)
        use_len = False
    else:
        tk_len = topk_length.to(torch.int32).contiguous()
        use_len = True
    if attn_sink is None:
        sink = torch.zeros(h_q, dtype=torch.float32, device=q.device)
        use_sink = False
    else:
        sink = attn_sink[:h_q].to(torch.float32).contiguous()
        use_sink = True
    runner = _get_kernel(
        h_q, d_qk, head_dim, topk, use_sink, use_len, float(scale),
    )
    # Direct-write fast path: caller already owns the bf16 (s_q, h_q,
    # head_dim) buffer in the right shape.  The kernel writes into it via
    # buffer_store, so the caller's tensor must be contiguous.
    direct_write = (
        output is not None
        and output.dtype == torch.bfloat16
        and output.shape == (s_q, h_q, head_dim)
        and output.is_contiguous()
        and output.device == q.device
    )
    if direct_write:
        runner(q.contiguous(), kv.contiguous(), indices2, tk_len, sink,
               output, s_kv, s_q)
        return None
    out = torch.empty(s_q, h_q, head_dim, dtype=torch.bfloat16, device=q.device)
    runner(q.contiguous(), kv.contiguous(), indices2, tk_len, sink, out, s_kv, s_q)
    if output is None:
        return out
    # Fallback: caller passed an output buffer that doesn't match our
    # write contract (different dtype, non-contiguous, etc.).  Copy with
    # implicit dtype cast.
    output.copy_(out)
    return None
