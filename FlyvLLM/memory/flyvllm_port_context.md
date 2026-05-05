---
name: FlyvLLM port project context
description: What FlyvLLM is, how it differs from FlyTileKernels and from upstream vLLM, what's ported, what's intentionally torch-fallback.
type: project
---
FlyvLLM is a separate Python package living next to FlyTileKernels under
`/app/flytilelang/tilelang-to-flydsl-skills/`.  Its purpose is to provide
drop-in FlyDSL replacements for the two TileLang kernels in
`vllm.model_executor.layers.mhc`:

- `mhc_pre(residual, fn, hc_scale, hc_base, ...) → (post_mix, comb_mix, layer_input)`
- `mhc_post(x, residual, post_layer_mix, comb_res_mix) → out`

**Why this is its own package, not part of FlyTileKernels:**
- Different API contract: vLLM's `mhc_pre`/`mhc_post` are inference-only
  (no autograd), and `mhc_pre` is a *fused* operation that bundles the
  RMS-norm + sigmoid splits + sinkhorn + apply_mix that FlyTileKernels
  exposes as separate ops.
- Different shape conventions: vLLM passes `post_layer_mix` with shape
  `(..., mhc, 1)` where FlyTileKernels uses `(..., mhc)`.  Keeping them
  separate avoids broadcasting confusion.
- Independent dependency: FlyvLLM should be pip-installable without
  pulling FlyTileKernels.  The kernels are duplicated (by design).

**How this maps to the upstream TileLang fused kernel:**

The TileLang `mhc_pre_big_fuse_tilelang` is a single kernel that splits
work between threads `< 32` (sinkhorn + post + comb) and threads `>= 32`
(pre + apply_mix).  Porting that exact thread-split scheme to FlyDSL is
non-trivial in this build (the AST rewriter handling of mid-block
barriers + per-warp role assignments would need careful work).  Instead,
FlyvLLM decomposes into four kernels chained in the host-side wrapper:

1. `tf32_hc_prenorm_gemm` — torch fallback (DeepGEMM is CUDA-only).
   Uses `torch.matmul` with `allow_tf32=True`.  Collapses split-K into a
   single full matmul + parks the result in slot 0.
2. `mhc_norm_split_kernel` — one thread per token, sums splits, computes
   RMS rsqrt, multiplies mixes, and splits into pre/post/comb_unnormed.
   Uses `flydsl.expr.math.exp` for the sigmoid implementation.
3. `sinkhorn_kernel` — one thread per token holds the full `(mhc, mhc)`
   state in registers; `mhc=4` ⇒ 16 fp32 registers per thread.
4. `pre_apply_mix_kernel` — one thread per `(token, h)` element with
   mhc unrolled in registers.

All four kernels are real FlyDSL kernels except the GEMM (torch fallback
because vendored hipBLASLt outperforms a hand-rolled MFMA kernel at
these sizes).

`mhc_post` is a single fused FlyDSL kernel — one thread per `(token, h)`
hidden-element pair, mhc unrolled in registers.

**How to apply:** When asked about FlyvLLM:
- Tests live in `tests/test_mhc_pre.py` and `tests/test_mhc_post.py`.
  All 15 tests pass on gfx950.
- The kernels reuse the same patterns documented in
  `~/.claude/projects/-app-flytilelang/memory/mhc_kernel_patterns.md`
  — specifically Patterns A (per-token elementwise), B (per-(token, h)),
  and C (register-resident sinkhorn).
- If the user asks for a deeply-fused single-kernel port, that's a known
  follow-up; the chained decomposition was chosen because it lands on
  reliable FlyDSL primitives in this build (no mid-kernel barriers
  across thread roles required).

**What is *not* ported:**
- The vLLM `direct_register_custom_op` integration.  FlyvLLM exposes
  plain Python functions; integrating them as torch custom ops for
  vLLM is a separate concern (the user's CLAUDE.md says vLLM
  integration is a downstream task).
- The fp8 / DeepGEMM scaled-mm path used by other vLLM layers.
- Backward passes (vLLM mhc.py is forward-only for inference).

**Invocation:**

```sh
cd /app/flytilelang/tilelang-to-flydsl-skills/FlyvLLM
pip install -e .
pytest tests/   # 15 passed
```
