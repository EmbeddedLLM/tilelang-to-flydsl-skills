# FlyTileKernels Port Status

Per-kernel state of the TileLang → FlyDSL port targeting **gfx950**.

**Honest framing.** This snapshot was produced in an environment without GPU
or build-system access; nothing has been compiled or executed.  Every kernel
listed as "ported" should be regarded as a *first-draft* that may still have
bugs the user will discover when running the upstream pytest suite on a real
gfx950 machine.  Stubs raise `NotImplementedError` with a one-line reason and
cannot be exercised at all.

## Legend

- ✅ **ported** — full FlyDSL implementation written; needs GPU validation.
- 🟡 **skeleton** — pseudocode-level skeleton in module docstring; the
  function raises `NotImplementedError` until the skeleton is filled in.
- ❌ **stub** — placeholder only; raises `NotImplementedError`.  No skeleton.

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
| `normalize_weight` | ✅ | Patterned exactly off skill's worked example; per-row L1 normalisation with 1e-20 bias. Verified vs `softmax_kernel.py` patterns. |
| `mask_indices_by_tp` | ❌ | Needs integer BufferCopy skeleton. |
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
| `cast_back` / `per_token_cast_back` | ❌ | fp8/fp4 dequantise back to bf16/fp32. |
| `per_token_cast_to_e5m6` / `cast_back_e5m6` | ❌ | Non-standard 1+5+6 fp format; bit-twiddling encode/decode. |

### transpose/

| Kernel | Status | Notes |
|---|---|---|
| `transpose` / `batched_transpose` | 🟡 | Skill repo has a ~150-line skeleton in `references/worked_examples/batched_transpose.md`. Needs verification of `SmemPtr.as_memref` + the decoded `loop_layout` thr/val pair. |

### mhc/

All MHC kernels are stubbed.  They are accessed only through the modeling
layer (`fly_tile_kernels.modeling.mhc.ops.*`), which itself imports them.
Calling any modeling-layer MHC op will surface the stub error.

### engram/

| Kernel | Status | Notes |
|---|---|---|
| `fused_weight` | ❌ | Engram fused-weight kernel. |
| `engram_gate_fwd` / `engram_gate_bwd` | ❌ | Engram gate fwd/bwd (fused with rmsnorm). |
| `grad_w_reduce` | ❌ | Engram weight gradient reduction. |
| `engram_hash` | ❌ | Engram hashing kernel. |

### modeling/

The modeling layer is pure-Python autograd Functions wrapping kernel calls.
Carried over verbatim with import rewrites; will work when the underlying
kernels do.

### testing/ + torch/

Both copied verbatim from upstream with `tile_kernels` → `fly_tile_kernels`
rewrites.  These are pure Python and have no FlyDSL dependency.

## Recommended order for completing the port

1. **Resolve the integer-atomic blocker** — exposes `group_count`, `aux_fi`,
   `get_fused_mapping`, `expand_to_fused` in one shot.
2. **Build the integer BufferCopy skeleton** — exposes `mask_indices_by_tp`,
   `inplace_unique_group_indices`, the int64 store for `topk_gate`.
3. **Port `topk_gate`** as a self-contained reduction-style kernel.
4. **Port the worked-example transpose** as a shared-memory exemplar.
5. **Port `cast_back`** as a simple quant exemplar (load, scale, store).
6. **Port `per_token_cast`** — the canonical per-row reduction-then-cast.
7. **Decide on the wave-size strategy** for `top2_sum_gate` and port it.
8. **Port the remaining quant + MHC + engram kernels** in any order;
   they share the per-token / per-block reduction skeleton.

## Testing strategy on gfx950

```sh
# Build and install FlyDSL first (see FlyDSL/scripts/build*.sh).
pip install -e ".[dev]"
cd FlyTileKernels

# Run only the ported kernel's tests:
pytest tests/moe/test_normalize_weight.py -n 4

# Try the full suite to see the stub failure pattern:
pytest tests/ -n 4
# Each unported kernel will raise NotImplementedError with the message
# "FlyTileKernels: `<name>` has not been fully ported... See STATUS.md".
```
