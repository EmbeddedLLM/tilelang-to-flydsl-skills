---
name: MHC port progress
description: Current state of the MHC (Manifold HyperConnection) kernel port — what's real FlyDSL, what's torch-fallback, and why each.
type: project
---
As of 2026-05-01, all 115 MHC correctness tests pass on gfx950 (8 benchmarks
skipped).  The port lands as a mix of real FlyDSL kernels (forward paths) and
torch fallbacks (most backwards + norm_fn's GEMM).

**Why:** The user wants real FlyDSL kernels for the MHC layer to feed into
vLLM integration.  Forward paths are perf-critical and have been ported.
Backwards are kept as torch autograd-replay fallbacks where the cross-block
reductions for partial-buffer accumulators (mhc_scale_grad, mhc_base_grad,
d_post_mix, d_comb_mix) are non-trivial and the test tolerance admits them.
norm_fn's matmul stays on `torch.matmul` because hipBLASLt outperforms
hand-rolled FlyDSL GEMMs at the test sizes (mhc_mult=4, hidden in [1280,
8192]).

**How to apply:** When asked to port MHC kernels, recognise that this batch
already landed and check `STATUS.md` in the FlyTileKernels root for the
current matrix.  Avoid duplicating effort.

## Per-kernel state (forward path / backward path)

| Kernel                       | Fwd      | Bwd      |
|------------------------------|----------|----------|
| `mhc.expand`                 | FlyDSL ✅ | FlyDSL ✅ |
| `mhc.head_compute_mix`       | FlyDSL ✅ | torch    |
| `mhc.pre_split_mixes`        | FlyDSL ✅ | torch    |
| `mhc.sinkhorn`               | FlyDSL ✅ | torch    |
| `mhc.pre_apply_mix`          | FlyDSL ✅ | torch    |
| `mhc.post`                   | FlyDSL ✅ | torch    |
| `mhc.multilayer_recompute`   | chained  | n/a      |
| `mhc.pre_big_fuse`           | composed | n/a      |
| `mhc.norm_fn` (all variants) | torch    | torch    |

- **`multilayer_recompute`** is implemented as a Python loop over real-FlyDSL
  `mhc_pre_apply_mix` + `mhc_post` calls (matches the unfused reference
  bit-exactly because both paths go through the same kernels).
- **`pre_big_fuse`** is a torch composition that calls into the FlyDSL
  sinkhorn forward (`_mhc_sinkhorn_fwd`) so its `comb_mix` matches the
  unfused reference under `torch.equal`.
- **`norm_fn`**: all five factory functions (`fwd_mul`, `fwd_norm`,
  `bwd_mul`, `bwd_norm`, `fn_normw_merge_fwd/bwd`) are torch fallbacks.
  Forward GEMM uses `torch.matmul` with `allow_tf32=True` to match the
  reference's tf32 accumulation.

## Backward-path blockers, if you decide to port

The torch fallbacks for bwd are correct but if you want real FlyDSL bwd
kernels, the cross-cutting blocker is **per-SM partial buffer reduction**:

- `head_compute_mix`, `pre_split_mixes`: need cross-block reductions for
  `mhc_scale_grad` (scalar, 1 element) and `mhc_base_grad` (mhc_mult or
  2*mhc + mhc^2 elements).  TileLang uses persistent kernels with
  `T.alloc_reducer(replication='all')` plus `T.finalize_reducer`.  In
  FlyDSL this needs either:
  - global atomics (float-add works via `raw_ptr_buffer_atomic_fadd`)
  - or per-SM partial buffers + host-side `.sum(0)` (the current modeling
    layer already expects partial buffers, so just zero-fill rows other
    than slot 0 and the wrapper still works — but writing all rows in
    parallel needs persistent grids or atomic-add).
- `pre_apply_mix.bwd`, `post.bwd`: need wave/block reduction over the h
  dimension for `mix_grad`/`d_post_mix`/`d_comb_mix`.  The wave-reduce
  butterfly in `_flydsl_helpers.py::wave_reduce` is the right primitive,
  but it must be paired with an LDS-stage block reduction since
  `block_reduce` requires a single-wave finaliser.
- `sinkhorn.bwd`: needs the recurrence-replay state stored across
  iterations (TileLang stages it in shared memory).

## Test invocation

```sh
cd /app/flytilelang/tilelang-to-flydsl-skills/FlyTileKernels
pytest tests/mhc/  # 115 passed, 8 skipped (benchmarks)
```

Some tests are slow (`pre_big_fuse` is ~50s) because every test compiles
and launches the FlyDSL sinkhorn kernel inside the iteration.
