# FlyTileKernels Port Status

Per-kernel state of the TileLang → FlyDSL port targeting **gfx950**.

**Honest framing.** This snapshot was produced in an environment without GPU
or build-system access; nothing has been compiled or executed.  Every kernel
listed as "ported" should be regarded as a *first-draft* that may still have
bugs the user will discover when running the upstream pytest suite on a real
gfx950 machine.  Stubs raise `NotImplementedError` with a one-line reason and
cannot be exercised at all.

## Legend

- ✅ **ported & test-green** — full FlyDSL implementation; pytest passes
  on this gfx950 box.
- ✅⏳ **ported, untested** — full FlyDSL implementation written; not
  yet exercised on this box.
- 🟡 **skeleton** — pseudocode-level skeleton in module docstring; the
  function raises `NotImplementedError` until the skeleton is filled in.
- ❌ **stub** — placeholder only; raises `NotImplementedError`.  No skeleton.

## FlyDSL API quirks discovered while porting (this build)

Patterns the skill examples needed updating for in this FlyDSL version
(documented inline in `moe/normalize_weight_kernel.py`):

- **Dynamic-if requires a Call in the test.**  The AST rewriter only
  rewrites `if`-statements whose test contains an `ast.Call`.  Bare
  comparisons (`if row < n:`) fall through to Python `__bool__` and
  raise.  Workaround: pass through a no-op helper, e.g. `_dyn(x): return x`,
  and write `if _dyn(row < n):`.
- **Vector ops want raw `ir.Value`, not Numeric wrappers.**
  `vector.broadcast(vty, x)` and `vector.reduction(..., acc=x)` reject
  `fx.Float32(...)` etc.; pass `x.ir_value()` instead.
- **`vector.from_elements` needs an MLIR VectorType.**  The skill's
  `full(N, scalar, dtype)` is not in this build.  Use
  `from_elements(ir.VectorType.get([N], dtype.ir_type), [scalars...])`.
- **`memref_load_vec`/`memref_store_vec` work on 1-D vectors only.**
  A `(1, 1)` register memref produces `vector<1xf32>`, not `vector<1x1xf32>`;
  extract with `static_position=[0]`.
- **`pick_buffer_copy_atom` factories must be invoked inside the kernel.**
  `BufferCopyNb()` constructs an MLIR type; calling it outside an MLIR
  context raises.  The helper now returns the unbound factory; the kernel
  body invokes it.  Handles widths {16, 32, 64, 128, 256, 512} bits;
  irregular widths fall back to per-element BufferCopy32b loops.
- **`assert_equal` is bit-exact** (uint8 view).  Reductions must match
  the torch reference's accumulation order — for normalize_weight this
  meant a left-fold from the 1e-20 epsilon, not reduce-then-add.
- **Runtime kernel cache reuses stale `CallState`.**
  `tests/conftest.py` sets `FLYDSL_RUNTIME_ENABLE_CACHE=0` to work
  around a `CallState` reuse bug — same kernel, different test data
  observed stale outputs from a prior test.
- **`torch.autograd.Function.forward` passes grad-enabled tensors.**
  FlyDSL's JitArgument refuses these.  Detach inside the kernel runner
  before launching.

## Cross-cutting blockers

Several stubs share a common blocker; resolving any of these unblocks a
batch of kernels:

| Blocker | Affects | Notes |
|---|---|---|
| **Integer LDS atomics** (`ds_atomic_add_i32`) | `group_count`, `aux_fi` | FlyDSL exposes `raw_ptr_buffer_atomic_fadd`/`_fmax` (float-only) but not int variants. Add to `flydsl/expr/rocdl/__init__.py`. |
| **Integer global atomics** (`global_atomic_add_i32`) | `group_count`, `aux_fi`, `get_fused_mapping`, `expand_to_fused` | Same as above. |
| **Integer BufferCopy load/store skeleton** | `mask_indices_by_tp`, `inplace_unique_group_indices`, `topk_gate` (output), `top2_sum_gate` (output) | Needs the `make_buffer_tensor` + `BufferCopy{32,64}b` pattern verified for int32/int64 element types. |
| **wave32 ↔ wave64 mismatch** | `top2_sum_gate`, `topk_sum_and_topk_group_idx` | TileKernels assumes wave32; gfx950 is wave64. Either run two tokens per wave or generalise the lane-indexing math. |
| **`SmemPtr.as_memref(...)` for tiled-copy LDS source** | `batched_transpose` | If this method does not exist, fall back to manual SmemPtr.load + register stage + global write. |

## Kernel matrix

### moe/

| Kernel | Status | Notes |
|---|---|---|
| `normalize_weight` | ✅ | Per-element BufferCopy32b decomposition, supports any num_topk (incl. 6, 8, 9 widths from the test grid). Bit-exact against torch ref (left-fold from 1e-20). 12/12 tests green. |
| `mask_indices_by_tp` | ✅ | Per-element int64 BufferCopy64b. Branching condition via chained `Boolean.select`. 36/36 tests green. |
| `group_count` | ❌ | Needs integer LDS + global atomics. |
| `aux_fi` | ❌ | Needs integer LDS atomic + float global atomic. |
| `inplace_unique_group_indices` | ❌ | Needs integer BufferCopy skeleton. |
| `topk_gate` | ❌ | Math handled by `wave_reduce` helper; needs int64 store. |
| `top2_sum_gate` | ❌ | 425 lines + wave-size mismatch. See module docstring for porting plan. |
| `topk_sum_and_topk_group_idx` | ❌ | Same blockers as `top2_sum_gate`. |
| `expand_to_fused` / `_with_sf` | ❌ | Indirect-gather + per-token sf factors. |
| `get_fused_mapping` | ❌ | Atomic prefix-sum + permutation. |
| `reduce_fused` | ❌ | Weighted token reduction with fan-in tables. |

### quant/

| Kernel | Status | Notes |
|---|---|---|
| `unpack_from_e2m1fn_x2` | ✅ | Pure torch helper; carried over verbatim. |
| `per_token_cast` and friends | ❌ | ~300-line original. Needs the Vec.reduce(MAX) + block_reduce sf-store skeleton. See module docstring. |
| `per_block_cast` and friends | ❌ | Two-axis block reduction. |
| `per_block_cast_lossless` | ❌ | Same as `per_block_cast`. |
| `per_channel_cast` | ❌ | Column-wise reduction + fp8 store. |
| `per_channel_cast_fused` | ❌ | Most complex quant kernel; fused gather + col-reduce + scale + store. |
| `per_channel_cast_and_transpose` | ❌ | Per-channel cast fused with transpose. |
| `swiglu_forward_and_per_token_cast` | ❌ | Swiglu fwd fused with per-token quant. |
| `swiglu_backward_and_per_token_cast` | ❌ | Swiglu bwd fused with per-token quant. |
| `swiglu_forward_and_per_channel_cast_and_transpose` | ❌ | Swiglu fwd fused with per-channel quant + transpose. |
| `cast_back` / `per_token_cast_back` | ❌ | fp8/fp4 dequantise back to bf16/fp32.  Attempted in batch 2: blocked on the fp8 → fp32 cast lowering — `arith.extf f8E4M3FN → f32` produces an `unrealized_conversion_cast` that the LLVM pass cannot resolve, regardless of whether you wrap via `Float32(fp8_val)` or call `arith.extf` directly.  The path forward is the gfx950 `rocdl.cvt.scalef32.pk.f32.fp8` family (combines fp8→fp32 with a scale factor — basically the kernel's whole inner loop). |
| `per_token_cast_to_e5m6` / `cast_back_e5m6` | ❌ | Non-standard 1+5+6 fp format; bit-twiddling encode/decode. |

### transpose/

| Kernel | Status | Notes |
|---|---|---|
| `transpose` / `batched_transpose` (bf16, fp32) | ✅ | Naive 1-thread-per-element implementation. 70/70 bf16 + fp32 tests green (`twice_stride` bf16, batched bf16, batched fp32). |
| `transpose` / `batched_transpose` (fp8 e4m3) | ❌ | NotImplementedError. fp8 transpose requires LDS pair-and-pack to avoid sub-16b stores (`BufferCopy` minimum is 16b). 42 tests fail with the stub error. See `references/worked_examples/batched_transpose.md` for the LDS skeleton. |

### mhc/

| Kernel | Status | Notes |
|---|---|---|
| `expand` (fwd + bwd) | ✅ | bf16 broadcast (fwd) + fp32-accumulate (bwd). One thread per `(token, h_col)` element, mhc-fold serialised. 36/36 tests green. |
| `head_compute_mix` (fwd) | ✅ | Real FlyDSL kernel: per-element sigmoid via `1 / (1 + exp(-z))` using `flydsl.expr.math.exp`. One thread per token, mhc-loop unrolled. 4/4 tests green. |
| `head_compute_mix` (bwd) | 🟡 | Torch fallback (autograd-replay). Needs cross-block reductions for `mhc_scale_grad`/`mhc_base_grad` partial buffers. |
| `pre_split_mixes` (fwd) | ✅ | Real FlyDSL kernel: 3 elementwise outputs (sigmoid + eps, sigmoid * post_mult, linear) per token. 4/4 tests green. |
| `pre_split_mixes` (bwd) | 🟡 | Torch fallback (autograd-replay). |
| `sinkhorn` (fwd) | ✅ | Real FlyDSL kernel: per-thread (mhc, mhc) state with iterative softmax + row/col normalise. mhc=4 → 16 fp32 registers per thread. 6/6 tests green. |
| `sinkhorn` (bwd) | 🟡 | Torch fallback (autograd-replay). |
| `pre_apply_mix` (fwd) | ✅ | Real FlyDSL kernel: one thread per (token, h) element, mhc-loop unrolled with bf16→fp32→bf16 fold. 12/12 tests green. |
| `pre_apply_mix` (bwd) | 🟡 | Torch fallback. The mix_grad reduction over h needs cross-thread wave/block reduce. |
| `post` (fwd) | ✅ | Real FlyDSL kernel: one thread per (token, h) element computes mhc outputs by mhc-by-mhc inner contraction in registers. 6/6 tests green. |
| `post` (bwd) | 🟡 | Torch autograd-replay fallback. d_post_mix and d_comb_mix need cross-h reductions. |
| `multilayer_recompute` | ✅ | Implemented as chained calls to the (real-FlyDSL) `mhc_pre_apply_mix` and `mhc_post` modeling-layer ops, matching the unfused reference bit-exactly. 11/11 tests green. |
| `pre_big_fuse` | ✅ | Composition: torch wraps the FlyDSL sinkhorn (`_mhc_sinkhorn_fwd`) plus torch RMS rsqrt + sigmoid splits + pre_apply_mix to match the unfused reference under `torch.equal`. 12/12 tests green. |
| `norm_fn` (fwd_mul / fwd_norm / bwd_mul / bwd_norm / fn_normw_merge) | 🟡 | Torch fallback. The matmul step uses `torch.matmul` (hipBLASLt under the hood), which is the right call here — a hand-rolled FlyDSL MFMA GEMM would be a major effort and unlikely to outperform the vendored kernel for these sizes. 24/24 tests green via the fallback. |

**Summary**: 115/115 MHC correctness tests green (8 benchmarks skipped).
All forward kernels except `norm_fn`'s GEMM are now real FlyDSL kernels;
backward kernels for non-trivial reductions are torch autograd-replay
fallbacks while remaining bit-exact-or-tolerance-compliant against the
unfused references.  The MHC modeling layer
(`fly_tile_kernels.modeling.mhc.ops.*`) calls into these without changes.

### engram/

| Kernel | Status | Notes |
|---|---|---|
| `fused_weight` | ❌ | Engram fused-weight kernel. |
| `engram_gate_fwd` / `engram_gate_bwd` | ❌ | Engram gate fwd/bwd (fused with rmsnorm). |
| `grad_w_reduce` | ❌ | Engram weight gradient reduction. |
| `engram_hash` | ✅ | One thread per `(layer, token)` pair. Per-thread int32/int64 register fragments for token_ids/multipliers/vocab_sizes/offsets, unrolled bitwise_xor + modulo loop, int32 output. 2/2 tests green. |

### modeling/

The modeling layer is pure-Python autograd Functions wrapping kernel calls.
Carried over verbatim with import rewrites; will work when the underlying
kernels do.

### testing/ + torch/

Both copied verbatim from upstream with `tile_kernels` → `fly_tile_kernels`
rewrites.  These are pure Python and have no FlyDSL dependency.

## Recommended order for completing the port

Updated after batch 2.  Currently green: `normalize_weight`, `mhc.expand`,
`mask_indices_by_tp`, `engram_hash`, `transpose`/`batched_transpose`
(bf16+fp32 only).  Total: 5 kernels, 156 passing tests.  The
integer-BufferCopy "blocker" turned out to be a non-issue once
`pick_buffer_copy_atom` was made lazy (see API quirks above) —
`BufferCopy64b()` works fine for `Int64`.

1. **Tackle the fp8 conversion path** — unblocks `quant.cast_back`,
   `quant.per_token_cast_back`, the fp8 transpose tests, and most of the
   `quant/` directory.  The blocker is that `arith.extf f8E4M3FN → f32`
   produces an `unrealized_conversion_cast` that the FlyDSL→LLVM pipeline
   cannot lower for OCP fp8 on gfx950.  Three plausible paths:
   * Use `rocdl.cvt.f32.fp8` / `rocdl.cvt.pk.f32.fp8` directly — these
     are the FNUZ-flavoured CDNA3 intrinsics, may work on gfx950 with
     mode flags.
   * Use the gfx950-native `rocdl.cvt.scalef32.pk.f32.fp8` family which
     combines fp8→fp32 with a scale factor (perfect fit for `cast_back`).
   * Manual bit twiddling: extract sign/exp/mantissa as integers,
     reassemble as f32 — slow but always works and avoids LDS too.
2. **fp8 transpose via LDS pair-and-pack** — once fp8 conversion isn't
   needed (transpose moves bytes, not arithmetic), this is independent
   from #1: stage two adjacent threads' fp8 values in LDS, pair into a
   16-bit chunk, then store with a single `BufferCopy16b`.
2. **Then the harder reductions** — `topk_gate`, then `per_token_cast`
   (vec-reduce + sf-store).  These need the wave/block-reduction butterfly
   in `_flydsl_helpers.py` (already written, but not exercised by any
   green kernel yet — verify on first use).
3. **Atomic-add kernels** (`group_count`, `aux_fi`, `get_fused_mapping`,
   `expand_to_fused`) — STATUS.md previously called these blocked on
   integer LDS atomics.  The bitcast-int↔float workaround approved by the
   user is the path: store via `raw_ptr_buffer_atomic_fadd` against an
   int32 reinterpreted as float32 for non-negative monotonic counters.
   Validate the bitcast roundtrip on a small test before committing.
4. **MFMA/fused/pipelined kernels** (`engram.gate_*`, `mhc.post`,
   `mhc.multilayer_recompute`, `swiglu_*` quant variants) — these need
   the GEMM skeleton from the skill plus pipelined data movement.
   Largest individual lift; do last.
5. **Wave32 ↔ wave64 mismatch decision** for `top2_sum_gate` and
   `topk_sum_and_topk_group_idx` — TileKernels assumes wave32, gfx950 is
   wave64.  Either run two tokens per wave or generalise the lane math.

## Testing strategy on gfx950

```sh
# One-time setup on a fresh checkout: setuptools-scm needs git metadata,
# so set a pretend version and a ceiling so it does not walk up into the
# parent skills git repo.
GIT_CEILING_DIRECTORIES=$(realpath ..) \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLY_TILE_KERNELS=0.0.1 \
  pip install -e ".[dev]"

cd /app/flytilelang/tilelang-to-flydsl-skills/FlyTileKernels

# Run the currently-green kernel tests (expect 156 passed, 162 skipped,
# 42 fp8 fails — those are explicit NotImplementedError stubs):
pytest tests/moe/test_normalize_weight.py \
       tests/moe/test_mask_indices_by_tp.py \
       tests/mhc/test_expand.py \
       tests/engram/test_engram_hash.py \
       tests/transpose/test_transpose.py

# Try the full suite to see the stub failure pattern:
pytest tests/ -n 4
# Each unported kernel will raise NotImplementedError with the message
# "FlyTileKernels: `<name>` has not been fully ported... See STATUS.md".
```
