---
name: FlyvLLM test strategy
description: How the FlyvLLM tests reference the math, what tolerance is expected, and why no test compares against the upstream TileLang kernel.
type: reference
---
The FlyvLLM tests compare against a **pure-torch reference** that
replays the same math the FlyDSL pipeline computes.  They do *not*
compare against the upstream TileLang kernel for two reasons:

1. **TileLang doesn't run on ROCm**, so we cannot exercise it on the
   gfx950 box.
2. The TileLang fused kernel and the chained FlyDSL kernels can have
   different last-bit accumulation orders — comparing them with
   `torch.equal` would fail spuriously on tiny numerical differences.

Instead, the torch reference for `mhc_pre` reproduces exactly the same
chain the FlyDSL backend uses:
1. `x.float() @ fn.T` with `allow_tf32=True` (matches `tf32_hc_prenorm_gemm`)
2. `x.float().square().sum(-1)` (matches the sqrsum)
3. `(sqrsum / K + rms_eps).rsqrt()`
4. `mixes = gemm * rsqrt`
5. Sigmoid splits with `+ pre_eps` and `* post_mult`
6. Same iterative sinkhorn algorithm
7. `(residual.float() * pre.unsqueeze(-1)).sum(-2).bfloat16()`

Tolerances in `test_mhc_pre.py`:
- `post_mix`, `comb_mix`: `rtol=1e-4, atol=1e-5` (fp32 throughout).
- `layer_input`: `rtol=1e-2, atol=1e-2` (bf16 with mhc=4 fold; the fold
  order in our kernel is left-fold, torch's `.sum(-2)` is tree-reduce —
  diffs in the bf16 round-off trail).

`test_mhc_post.py` uses `assert_close` defaults (no explicit tolerance),
which is sufficient since the kernel matches the einsum + add reference
within bf16 precision.

**Why:** The chained reference catches porting bugs (wrong index, wrong
sigmoid path, wrong sinkhorn iteration count) without requiring any
external GPU truth table, and is robust to bit-level reduction-order
differences between FlyDSL and the unfused reference.

**How to apply:** When adding new FlyvLLM tests, follow the same
pattern: write a torch implementation of the exact algorithm the FlyDSL
kernel implements, parameterise over the test grid (n_tokens × hidden),
and compare with `torch.testing.assert_close` at the right tolerance.
