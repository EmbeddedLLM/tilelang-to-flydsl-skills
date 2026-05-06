# FlyvLLM kernels — benchmarks

Measured on a single gfx950 (MI355X) GPU, FlyDSL 0.1.2, torch 2.10 nightly,
median of 25 iterations after 5 warm-up calls (`bench()` from
`fly_vllm/kernels/_bench.py`).  `compile` is `torch.compile(dynamic=False)`
of the torch reference; it is skipped at d_qk=576 because the warm-up
dominates the wall-clock at those sizes.

## sparse_attn_prefill

FlashAttention-2 style sparse-attention prefill — gathers KV at top-k
indices, runs an online softmax, optionally folds an attention sink and a
per-query top-k length cap.  Kernel: `sparse_attn_prefill.py`.  Tests +
benchmark harness: `tests/test_sparse_attn.py`.

Public entry point (matches the vLLM
`vllm.v1.attention.ops.rocm_aiter_mla_sparse.rocm_sparse_attn_prefill`
contract — the call site at
`vllm.model_executor.layers.deepseek_v4_attention` line 1026):

```python
rocm_ref_sparse_attn_prefill_flydsl(
    q, kv, indices, topk_length, scale, head_dim, attn_sink,
    output=output_buf,   # pre-allocated bf16 (s_q, h_q, head_dim) — written in place
)                        # returns None
```

`output` is optional.  When omitted (the test-harness style), a fresh
bf16 tensor is allocated and returned.

Re-run with:

```sh
cd tilelang-to-flydsl-skills/FlyvLLM
python -m pytest tests/test_sparse_attn.py::test_sparse_attn_prefill_benchmark -v -s
```

### Latest results (2026-05-06, after exp2 + log2e fold + direct-write API)

| config          | shape (s_q, s_kv, h_q, d_qk, topk) | torch     | torch.compile | flydsl alloc-and-return | flydsl direct-write | speedup vs torch |
|-----------------|------------------------------------|-----------|---------------|-------------------------|---------------------|------------------|
| v3_h8_topk256   | 64, 1024,  8, 192,  256            | 312.80 us | 101.88 us     |  53.36 us               |  50.92 us           | 6.14×            |
| v4flash_tp4     | 64, 2048, 16, 576,  512            | 285.76 us | n/a           | 139.36 us               | 136.28 us           | 2.10×            |
| v4pro_tp8       | 64, 4096, 16, 576, 1024            | 278.84 us | n/a           | 255.40 us               | 247.56 us           | 1.13×            |

(Medians of `bench` invocations, each median-of-25 timed iters after
5 warm-ups.)  The direct-write column is the path used by
`vllm.model_executor.layers.deepseek_v4_attention` (line 1026), where
the caller passes a pre-allocated bf16 ``output`` buffer; the kernel
writes into it in place and the wrapper returns ``None``.

`v4pro_tp8` is at the HBM-bandwidth limit (~240 us for the ~1.18 GB total
gather across all blocks at ~5 TB/s).  Beating it further would require
sharing gathered KV across heads (grid-restructure) or MFMA-based matmul.

### Optimization log

Last optimization round transplanted ideas from the upstream
`FlyDSL/kernels/mla_fwd_decode_m16x8_fp8_fp8.py` reference kernel.  Each
candidate was gated on correctness + bench improvement; only winners are
in the kernel.

| candidate                                              | result                              |
|--------------------------------------------------------|-------------------------------------|
| `rocdl.exp2` (instead of `fxmath.exp`) + log2e fold into `softmax_scale` | **kept** — V3 −5%, V4 flat (no regression) |
| `vector.store` (single ds_write_b128) in V4 gather     | rejected — LLVM was already coalescing the 8 ds_write_b16; flat |
| `fxmath.fma` to force FMA fusion in V4 score + output loops | rejected — regressed V4 by 20–25% (apparently inhibits unrolling/scheduling) |
| Grid swap `(s_q, h_q)` → `(h_q, s_q)` for L2 reuse across heads | rejected — regressed V4pro by ~16% (the original ordering benefits more from spreading reads than the swap does from L2 reuse) |

### Optimization stack (in order of impact)

1. **128-bit vectorized buffer-copy gather** (`BufferCopy128b` → 8 bf16 per
   HBM transaction).  8× fewer buffer-load instructions; the dominant V4
   win, also gives a ~1.7× V3 speedup over a scalar gather.
2. **Runtime gather-loop**: keeps the per-tile gather body unrolled for
   ILP, but iterates the tile loop with `scf.for` so the V4 18-pass × 32-tile
   product (=576 unrolled passes) never reaches LLVM.
3. **Runtime tile loop with iter args** (`m`, `l`, `o_regs[0..3]` threaded
   through `scf.for`).  Without this the V4 NUM_TILES=32 unroll hangs the
   LLVM compile.
4. **4-thread-per-score collaboration** (V4 only): when BLOCK_N=32 < 128
   threads, split each score's dot product across
   `THREADS_PER_SCORE = NUM_THREADS / BLOCK_N` lanes and merge with a
   partial-wave butterfly.  V3 keeps the simpler 1-thread-per-score path.
5. **`(yield expr)`-as-expression capture** of for-op results — avoids a
   per-iteration register-memref round-trip and lets LLVM keep the carried
   accumulators in VGPRs.
6. **Runtime n-loop in the output update** (V4 only): carries `o_regs`
   through `scf.for` to avoid 4096 unrolled FMAs at v4pro size.
7. **`rocdl.exp2` + log2e baked into `softmax_scale`**: the per-tile
   softmax runs in the log2 domain so `exp(score - m)` collapses to a
   single `v_exp_f32` (`rocdl.exp2`) with no per-call log2e multiply.
   `sink_val` is pre-multiplied by log2e at load.

### Configuration table

| name | d_qk regime | BLOCK_N | score d-loop  | THREADS_PER_SCORE |
|------|-------------|---------|---------------|-------------------|
| V3   | ≤ 256       | 64      | constexpr     | 2                 |
| V4   | > 256       | 32      | runtime scf.for | 4               |

`BLOCK_N=32` for V4 is forced by the gfx9 64 KB per-block LDS budget:
`BLOCK_N × d_qk × 2 B` is 73 KB at BLOCK_N=64, d_qk=576 — overflows.
