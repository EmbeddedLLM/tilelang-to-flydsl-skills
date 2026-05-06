# sparse_attn_prefill — performance report

Roofline analysis and optimization plan for
`fly_vllm/kernels/sparse_attn_prefill.py`, captured after the 2026-05-06
optimization round.  Audience: anyone picking this kernel up later to push
performance further.

Current status, gfx950 (MI355X), FlyDSL 0.1.2:

| config         | shape (s_q, s_kv, h_q, d_qk, topk) | flydsl   | torch   | speedup |
|----------------|------------------------------------|----------|---------|---------|
| v3_h8_topk256  | 64, 1024,  8, 192,  256            |  54.7 us | 312 us  | 5.7×    |
| v4flash_tp4    | 64, 2048, 16, 576,  512            | 140.1 us | 286 us  | 2.0×    |
| v4pro_tp8      | 64, 4096, 16, 576, 1024            | 253.0 us | 279 us  | 1.10×   |

DSv4 is the production target.

---

## 1. Per-block work (the AI calculation)

For each `(sq, hq)` block:

- **Score (QK)**: Q · K_gathered → `2 · topk · d_qk` flops
- **Output (PV)**: P · V_gathered → `2 · topk · head_dim` flops
- **HBM read**: `topk · d_qk · 2` bytes (KV gather; everything else — Q,
  indices, sink, output write — is small enough to ignore)

So the arithmetic intensity is

```
AI  =  (2·topk·d_qk + 2·topk·head_dim) / (topk·d_qk·2)
    =  (d_qk + head_dim) / d_qk         [flops per byte]
```

| config | d_qk | head_dim | **AI** |
|--------|------|----------|--------|
| V3     | 192  | 128      | 1.67   |
| V4     | 576  | 512      | **1.89** |

Both extremely low.  At gfx950's fp32 roofline ridge (~33 flops/byte =
163 Tflops / 5 TB/s) we are 17–18× below; at the bf16-MFMA ridge
(~260 flops/byte = 1.3 Pflops / 5 TB/s) we are 138× below.  **This
kernel is fundamentally HBM-bandwidth-bound on the current algorithm.**
No amount of compute-side tuning will move the needle once we are
saturating BW.

## 2. Achieved efficiency (median of 5 runs)

Total per kernel invocation = `s_q · h_q · per-block`.

| config         | flops total  | bytes total | time     | achieved compute | achieved BW   | % of 5 TB/s |
|----------------|--------------|-------------|----------|------------------|---------------|-------------|
| v3_h8_topk256  | 0.084 Gflops | 0.048 GB    |  54.7 us | 1.54 Tflops      | 0.88 TB/s     |  18%        |
| v4flash_tp4    | 1.14 Gflops  | 0.604 GB    | 140.1 us | 8.14 Tflops      | **4.31 TB/s** | **86%**     |
| v4pro_tp8      | 2.28 Gflops  | 1.18 GB     | 253.0 us | 9.01 Tflops      | **4.66 TB/s** | **93%**     |

vs gfx950 fp32 peak (~163 Tflops): **~5%** compute efficiency on V4.  This
is exactly what the roofline predicts: at AI≈1.89 and 5 TB/s, the *most*
sustainable compute is `1.89 · 5 = 9.5 Tflops`, and we are hitting that.

V3's 18% BW figure is misleading — its working set (48 MB total) is too
small to saturate HBM, so it operates closer to the L2/LDS-latency regime
than the BW-roofline regime.  At 5.7× torch it is unlikely to be the
optimization target.

## 3. Why v4pro can't go much faster on the current algorithm

v4pro is at **93% of 5 TB/s**, leaving ~17 us of HBM-side headroom and
nothing on the compute side.  Pushing further needs a structural change
that *reduces HBM bytes per useful flop* — i.e. raises the AI.

### 3.1 Two avenues for raising AI

**Option A — Cross-head KV reuse (highest value, biggest rewrite)**

Indices depend only on `sq`, not `hq`.  Today each `(sq, hq)` block
re-gathers the same KV rows from HBM independently.  With `h_q=16`,
**every KV byte is read 16×**.

If the kernel processed all 16 heads for a given `sq` together in one
work-item (either via grid restructure to `(s_q,)` with internal head
loop, or a persistent-thread design), HBM bytes drop 16× and AI jumps to
`16 × 1.89 ≈ 30 flops/byte`.  That is *past* the fp32 ridge — V4 would
become compute-bound, and an MFMA implementation would then matter.

Time floor estimate for v4pro under cross-head reuse:
- HBM: `1.18 GB / 16 / 5 TB/s ≈ 15 us`
- Compute (fp32 scalar muladd at ~10 Tflops sustained): `2.28 Gflops / 10 Tflops ≈ 230 us`
- Compute (bf16 MFMA at ~25% of 1.3 Pflops sustained): `2.28 Gflops / 325 Tflops ≈ 7 us`

So **with cross-head reuse + MFMA**, v4pro could plausibly drop from
253 us → tens of microseconds.  Without MFMA, cross-head reuse alone
makes the kernel compute-bound on fp32 scalar (no win — possibly a
regression vs the BW-bound version that uses 100% of BW).

**Conclusion**: cross-head reuse and MFMA are a *package*.  Either
alone is unlikely to help; together they are the way to multi-X speedup.

**Option B — Cooperative gather across blocks via L2 (low effort, low ceiling)**

Try to land enough cross-block KV reuse in the L2 cache that some HBM
traffic is replaced by L2 hits.  The simple grid swap was tried (see
§4.4) and regressed.  Other variants worth trying:
- Two-level tile: launch grid `(s_q,)` with each block iterating its
  own `h_q`, but keep KV in LDS / registers across heads.
- Adjust grid stride so co-scheduled blocks share `sq` *without*
  starving the memory subsystem.

Ceiling for option B alone is at most `1 - 1/16 = 94%` HBM reduction,
which is the same as option A on the bandwidth side.  Without MFMA we
hit the same fp32-compute floor.

### 3.2 Why "more compute optimization" doesn't help v4pro

Things that *would* help if we were compute-bound:
- Force FMA fusion in the score/output loops
- Vectorized LDS reads (ds_read_b64) for the score d-loop and PV n-loop
- LDS bank-conflict-free K layout

…but at 93% of HBM peak, they're invisible (or negative — see §4.3 for
why fxmath.fma actually regressed).  These become valuable *after*
option A reduces HBM pressure enough to expose compute.

## 4. Optimization log (what's been tried)

### 4.1 Landed wins (already in the kernel)

In rough order of impact on V4:

1. **128-bit vectorized buffer-copy gather** (`BufferCopy128b` → 8 bf16
   per HBM transaction).  8× fewer buffer-load instructions; the
   dominant V4 HBM win.  Also 1.7× V3 over a scalar gather.
2. **Constexpr-unrolled gather body inside a runtime tile loop**.
   V4 gather has 18 passes/tile × 32 tiles = 576 passes; fully unrolling
   this overflows LLVM, fully runtime-looping it loses ILP.  The split
   gives a single-tile-body unrolled kernel.
3. **Runtime tile loop with `m, l, o_regs[0..3]` as iter-args**.  Bounds
   the unrolled basic block to one tile body.  Without it, NUM_TILES=32
   hangs the LLVM compile.
4. **4-thread-per-score collaboration** (V4 only, BLOCK_N=32 < 128).
   Splits each score's dot product across 4 lanes via partial-wave
   butterfly.  Without it 75% of threads are idle in the score compute.
5. **`(yield expr)`-as-expression capture** of for-op results — keeps
   carried accumulators in VGPRs instead of round-tripping through
   register memrefs.
6. **Runtime n-loop in the output update** (V4 only).  Avoids
   `BLOCK_N · OUT_PER_THREAD · NUM_TILES = 4096` unrolled FMAs at
   v4pro size.
7. **`rocdl.exp2` + log2e folded into `softmax_scale`**.  Per-tile
   softmax now runs in the log2 domain so `exp(score - m)` is one
   `v_exp_f32` with no per-call `*log2e`.  V3 −5%, V4 within noise.

### 4.2 K_SMEM_PAD (V4-only)

`k_smem` row stride padded by 2 bf16 elements when the natural
`(DK · 2 / 4) % 32 == 0` would create a 32-way bank conflict.
Empirically:
- V4 (DK=576, runtime d-loop): +2% with padding → kept.
- V3 (DK=192, constexpr-unrolled inner loop): **−60%** with padding
  (LLVM scheduling appears to dislike the non-power-of-2 stride).

Hence the `USE_RUNTIME_D_LOOP and …` guard on the pad.

### 4.3 Tried + rejected (from the most recent round)

| candidate                                                          | result                                                                  |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| `vector.store` (single ds_write_b128) for the 8 bf16 LDS stores in the gather inner | flat — LLVM was already coalescing the 8 ds_write_b16 into a wide store |
| `fxmath.fma` to force FMA fusion in the V4 score d-loop and output n-loop | **−20–25% on V4** — likely inhibits LLVM unrolling/scheduling.  The default `+`/`*` on `fx.Float32` produces `mul + add` chains that LLVM unrolls and pipelines well.  Forcing `math.fma` collapses each iteration to a serialized FMA on the same accumulator, defeating the pipelining. |
| Grid swap `(s_q, h_q)` → `(h_q, s_q)` for L2 reuse across heads | **−16% on v4pro** — the original ordering spreads reads across different KV regions; the swap tries to exploit L2 reuse but loses more to memory-subsystem contention than it gains.  Naive grid restructure isn't enough; option A (§3.1) needs an *algorithmic* change that makes a single workitem own multiple heads. |

### 4.4 Things deliberately not tried (compute-side, bounded by §3.2)

- MFMA-based score/output GEMMs.  Single-row Q per block doesn't map.
  Would require batching queries (i.e. option A) first.
- Double-buffered KV LDS with async prefetch.  V4 single-buffer already
  uses 36 KB; doubling to 73 KB exceeds the 64 KB gfx9 per-block LDS
  budget at d_qk=576.  Could be made to fit with `BLOCK_N=16`, but that
  doubles the tile count and the per-tile fixed cost (barriers, softmax
  reductions, etc.).
- `s_setprio` / `sched_barrier` interleaving.  Most of the value of
  these in the reference MLA decode kernel comes from packing MFMA
  bursts; without MFMA they're scheduling hints with marginal value.
- Persistent-thread / work-loop dispatch.  Useful only if launch
  overhead were significant; ours is dwarfed by the 100+ us per-call
  HBM time.

## 5. Concrete next-steps if you decide to push v4pro further

In order of value:

1. **Cross-head KV reuse + MFMA together** (option A + MFMA).  Big
   rewrite: grid becomes `(s_q,)`, kernel iterates heads internally
   (or in registers), QK and PV compute via bf16 MFMA atoms with an
   MFMA-friendly LDS layout.  Reference idioms in
   `FlyDSL/kernels/mla_fwd_decode_m16x8_fp8_fp8.py`:
   - `KvManagerV2` LDS layout (per-warp 4-row sub-blocks with 2-DW pad)
   - `_async_load_k_tile` / `_prefetch_k_tile_asm` (inline-asm
     `buffer_load_dword lds` for direct VRAM→LDS without register hop)
   - `_load_k_from_lds` / `_load_v_from_lds` (strided ds_read_b64
     patterns matched to MFMA matrix-A layout)
   - `_warp_reduce_max_16` / `_warp_reduce_add_16` (16-lane butterfly
     reductions to match MFMA column groups)
   Plausible ceiling: 30–60 us for v4pro (vs current 253 us).

2. **fp8 KV path** (orthogonal to #1).  KV at fp8 halves the HBM bytes
   and doubles the AI to ~3.8 flops/byte.  Useful even on the existing
   algorithm.  Requires inputs already in fp8 format and an
   `mfma_*_fp8_fp8` (or scalar fp8→fp32 dequant inside the score/output
   loops, which would push compute pressure way up).  The reference
   kernel uses this for MLA decode.

3. **Cooperative L2 prefetch only** (option B).  Lower ceiling than
   option A, but doesn't require MFMA.  Probably a few percent at best.
   Worth measuring with proper hardware profiling before investing.

What **not** to spend time on without first having #1:

- Hand-scheduled `s_setprio` / `sched_barrier` / inline-asm waitcnt.
  These are valuable inside MFMA loops; without MFMA they tune ~0.
- Vectorized LDS reads in the score d-loop.  Same reason — score-loop
  compute is far below HBM, vectorizing it doesn't move the kernel
  time.
- Forcing FMA via `math.fma`.  Demonstrably regresses (§4.3).

## 6. Reference numbers you'll want during measurement

- gfx950 (MI355X) **theoretical** HBM3E peak: ~8 TB/s (8 stacks × 1 TB/s).
- gfx950 **realistic** kernel-achievable BW: ~5 TB/s (60–65% of peak,
  matching what we see in this kernel and other BW-heavy FlyDSL kernels).
- gfx950 fp32 vector peak: ~163 Tflops.
- gfx950 bf16 MFMA peak: ~1.3 Pflops dense.
- gfx9 per-block LDS budget: 64 KB.
- Bench harness: `tests/test_sparse_attn.py::test_sparse_attn_prefill_benchmark`,
  median of 25 timed iterations after 5 warm-ups; `_bench.py` enables
  the FlyDSL runtime kernel cache for fair apples-to-apples timing.
