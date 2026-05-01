---
name: FlyDSL API quirks vs the skill examples
description: Concrete differences between the skill's reference patterns and the FlyDSL 0.1.2 build installed at /usr/local/lib/python3.12/dist-packages/flydsl. Discovered while porting normalize_weight, mhc.expand, and mask_indices_by_tp.
type: reference
originSessionId: c032df3a-5434-4761-a4fd-491e1977a619
---
The `tilelang-to-flydsl` skill's reference port (`moe/normalize_weight_kernel.py`)
was written without ever being run on a real FlyDSL toolchain.  Several of its
patterns are out of date relative to FlyDSL 0.1.2 (the version pip-installed
on the gfx950 box at `/usr/local/lib/python3.12/dist-packages/flydsl`).  When
porting new kernels in this codebase, prefer the substitutes below.

**Why:** The first validation batch had to debug each pattern in turn
(repro: April 2026, /app/flytilelang).  Documenting these saves the next
session re-discovery time.

**How to apply:** Pattern-match new kernels against the working ports in
`fly_tile_kernels/{moe/normalize_weight_kernel.py, moe/mask_indices_by_tp_kernel.py,
mhc/expand_kernel.py}` rather than the skill's worked examples.

## Concrete substitutions

| What the skill says | What this build needs |
|---|---|
| `if cond:` for runtime conditions | The AST rewriter only treats `if` as dynamic when the test contains an `ast.Call`.  Wrap a comparison via a passthrough: `def _dyn(x): return x` and `if _dyn(row < n):`. |
| `v.reduce(ReductionOp.ADD)` on a Vector | `vector.reduction(dest=Float32.ir_type, kind=vector.CombiningKind.ADD, vector=v, acc=...)` — and `acc` must be a raw `ir.Value`, not a `Numeric` (use `.ir_value()`). |
| `full(N, scalar, dtype)` for vector construction | `vector.from_elements(ir.VectorType.get([N], dtype.ir_type), [scalars...])`.  `from_elements` auto-unwraps Numeric scalars. |
| `vector.broadcast(vty, scalar)` with a Numeric | Pass `scalar.ir_value()` — broadcast does NOT auto-unwrap. |
| `memref_load_vec(reg)` for a `(1, 1)` memref | Returns `vector<1xf32>` (1-D), not `vector<1x1xf32>`.  Extract scalar with `vector.extract(v, static_position=[0])` (single-element list, not `[0, 0]`). |
| `pick_buffer_copy_atom(...)` returning a constructed CopyOp | Refactored: returns the unbound factory + n_atoms.  Caller invokes the factory **inside** the `@flyc.kernel` body.  The original eager construction raises "MLIR function requires a Context" outside an MLIR context. |
| Multi-atom widths (n_atoms > 1) | The reference port raises NotImplementedError.  Workable substitute: per-element `BufferCopy32b` decomposition in a `range_constexpr(num_topk)` loop.  Slower but works for any element count and naturally matches torch's left-fold accumulation order. |
| `assert_equal` is allclose | It's bit-exact (uint8 view).  Reductions must match the torch reference's accumulation order — typically left-fold from epsilon, not reduce-then-add. |
| `torch.autograd.Function.forward` passes raw tensors | FlyDSL JitArgument refuses grad-enabled tensors.  Detach inside the kernel runner before calling the launcher. |
| Runtime kernel cache "just works" | There's a stale-`CallState` reuse bug: the second test with the same kernel signature but different tensor data observes outputs from the first.  `tests/conftest.py` defaults `FLYDSL_RUNTIME_ENABLE_CACHE=0` to work around it. |

## Slicing pattern that works

The skill's "just `slice` a 4-D `logical_divide`" pattern crashes inside
FlyDSL with rank-mismatch asserts.  The two-step pattern that DOES work
for per-element access on a 2-D buffer:

```python
row_view = fx.slice(buf, (token, None))         # 1-D length=cols
row_div  = fx.logical_divide(row_view, fx.make_layout(1, 1))
elem_view = fx.slice(row_div, (None, k))        # single element
fx.copy_atom_call(copy_atom_32, elem_view, reg)
```

For a 1-D buffer, single `make_layout(1, 1)` divide + `slice(div, (None, idx))`.
