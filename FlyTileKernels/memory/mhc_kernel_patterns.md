---
name: MHC kernel patterns
description: Recurring FlyDSL patterns used across the MHC ports (per-token elementwise, register-resident sinkhorn, per-(token, h) reductions). Use as a copy-paste guide for similar shapes.
type: reference
---
The MHC kernels share a small set of layout patterns, all distinct from the
patterns used in the moe/transpose kernels.  Document them here so the next
session doesn't re-derive them.

**Why:** Reading the MHC kernel sources, every fwd is a slight variation on
one of three layouts.  Rec ognising the pattern up front saves derivation
time.

**How to apply:** Match the new kernel's input/output shapes against the
patterns below; pick the closest layout and adapt.

## Pattern A: per-token elementwise (mhc inner-unrolled)

Use when output is `(n, mhc, ...)` or `(n, mhc^2)` and `mhc` is small (4 in
tests).

- Grid: `(ceil(num_tokens / NUM_THREADS), 1, 1)` with `NUM_THREADS=128`.
- Block: `(NUM_THREADS, 1, 1)`.
- Per-thread: one token; inner loop over mhc indices via `range_constexpr`.
- All scalar arithmetic uses per-thread fp32 register allocas + extracts
  (the `(1, 1)` register memref → `vector<1xf32>` → `vector.extract` chain
  documented in `flydsl_api_quirks`).

Examples: `head_compute_mix.fwd`, `pre_split_mixes.fwd`.

The sigmoid implementation `1 / (1 + exp(-z))` uses
`flydsl.expr.math.exp(neg_z)` — there is no built-in sigmoid, but math.exp
exists and lowers cleanly on gfx950.

## Pattern B: per-(token, h) elementwise/reduction (mhc inner-unrolled)

Use when output is `(n, h)` or `(n, mhc, h)` with `h` in {1280, ..., 8192}.

- Grid: `(num_tokens, ceil(h / BLK_H), 1)` with `BLK_H=256`.
- Block: `(BLK_H, 1, 1)`.
- Per-thread: one `(token, h_idx)` element.
- Per-token data (mix vector, comb matrix, post mix) is loaded *redundantly*
  by every thread of the block.  L1 cache covers this — for `mhc=4` we
  spend O(16-20) fp32 loads per thread on per-token data which is small
  vs the per-h global I/O.

Examples: `pre_apply_mix.fwd`, `post.fwd`.

bf16 ↔ fp32 conversion: load via `BufferCopy16b` into a bf16 register
memref, extract scalar, wrap with `fx.BFloat16(...)`, then `.to(fx.Float32)`
for arithmetic.  Reverse for stores.

## Pattern C: register-resident sinkhorn (per-thread mhc x mhc state)

Use when each token's data fits in O(mhc^2) fp32 registers.  For `mhc=4`,
that's 16 fp32 registers per thread — comfortable.

- Grid: `(ceil(num_tokens / NUM_THREADS), 1, 1)`.
- Block: `(NUM_THREADS, 1, 1)`.
- Per-thread: load all `H*H` values into a Python `vals[j][k]` 2-D list of
  `Float32` numerics; perform iterative softmax + row + column normalise
  with constexpr loops; store back.
- The Python list-of-numerics structure unrolls cleanly because every
  index is a constexpr; the compiler sees plain SSA values.

Example: `sinkhorn.fwd` (only kernel using this pattern).

## Pattern D: chained ops (no real FlyDSL kernel)

When a kernel is just a sequence of other ported kernels, write a Python
runner that calls them in order through the modeling-layer wrappers.  The
test's bit-exactness check (`torch.equal`) holds because both paths go
through the same backing kernels.

Example: `multilayer_recompute` (chains `mhc_pre_apply_mix` and `mhc_post`
through their `out=` parameters).

## Pattern E: torch-fallback (autograd-replay)

For backward kernels where the cross-block reduction is non-trivial and
the test allows `assert_close` tolerance, use:

```python
with torch.enable_grad():
    x_ = x.detach().requires_grad_(True)
    out = forward_torch(x_, ...)
    (grad,) = torch.autograd.grad(out, x_, grad_output)
return grad
```

This pattern is correct, fast enough for tests, and avoids the partial-
buffer accumulator + persistent-kernel dance.

Examples: every MHC bwd except `expand.bwd`.

## Don't forget: detach gradient-enabled tensors at the runner boundary

`torch.autograd.Function.forward` may pass tensors with `requires_grad=True`.
FlyDSL's `JitArgument` refuses these.  Every runner should call
`.detach()` on each torch.Tensor argument before invoking the launcher
(see any of the ported MHC fwd runners).
