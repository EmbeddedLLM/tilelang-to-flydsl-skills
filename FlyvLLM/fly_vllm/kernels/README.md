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

Re-run with:

```sh
cd tilelang-to-flydsl-skills/FlyvLLM
python -m pytest tests/test_sparse_attn.py::test_sparse_attn_prefill_benchmark -v -s
```

### Latest results (2026-05-06)

| config          | shape (s_q, s_kv, h_q, d_qk, topk) | torch     | torch.compile | flydsl    | speedup vs torch |
|-----------------|------------------------------------|-----------|---------------|-----------|------------------|
| v3_h8_topk256   | 64, 1024,  8, 192,  256            | 318.28 us | 102.44 us     |  57.52 us | 5.53×            |
| v4flash_tp4     | 64, 2048, 16, 576,  512            | 297.56 us | n/a           | 141.24 us | 2.11×            |
| v4pro_tp8       | 64, 4096, 16, 576, 1024            | 287.20 us | n/a           | 256.20 us | 1.12×            |

`v4pro_tp8` is at the HBM-bandwidth limit (~240 us for the ~1.18 GB total
gather across all blocks at ~5 TB/s).  Beating it further would require
sharing gathered KV across heads (grid-restructure) or MFMA-based matmul.

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

### Configuration table

| name | d_qk regime | BLOCK_N | score d-loop  | THREADS_PER_SCORE |
|------|-------------|---------|---------------|-------------------|
| V3   | ≤ 256       | 64      | constexpr     | 2                 |
| V4   | > 256       | 32      | runtime scf.for | 4               |

`BLOCK_N=32` for V4 is forced by the gfx9 64 KB per-block LDS budget:
`BLOCK_N × d_qk × 2 B` is 73 KB at BLOCK_N=64, d_qk=576 — overflows.
