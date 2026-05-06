# sparse_attn_prefill — vLLM integration debug runbook

State as of 2026-05-06.  The FlyDSL kernel works in standalone tests but
crashes inside vLLM serving DeepSeek-V4-Flash on TP=4.  This doc captures
where we are so the investigation can resume later.

## Current symptom

vLLM (`deepseek-ai/DeepSeek-V4-Flash`, TP=4 on gfx950) crashes after a
few prefill+decode iterations with:

```
!!!!!!! Segfault encountered !!!!!!!
  File "<unknown>", line 0, in 0xffffffffffffffff
```

(repeated once per TP worker — 4 lines).  No Python traceback because
the crash is below Python's frame.  Engine then dies with the standard
"RuntimeError: cancelled" downstream symptom (worker IPC).

The *first* failure mode (now fixed) was a deterministic FlyDSL MLIR
assertion:

```
intTupleSlice: Assertion `coord.rank() == tuple.rank() &&
                "Mismatched ranks in slice"' failed.
```

Cause: vLLM passes `kv=kv.view(-1, 1, q.shape[-1])` (3-D) but the kernel
was written for 2-D kv.  Fixed by squashing the spurious head axis in
`rocm_ref_sparse_attn_prefill_flydsl` (and defensively in the torch
reference).  See `_make_prefill_inputs`-style regression test
`test_sparse_attn_prefill_vllm_call_shape` in `tests/test_sparse_attn.py`.

The *second* failure (the segfault) appears *after* one or more
successful prefill calls — i.e. the kernel runs, returns, and something
later corrupts.  No reproducer yet; we don't know whether it's:

1. a kernel-side OOB write that taints downstream tensors,
2. a FlyDSL runtime cache / `CallState` bug (gfx950 0.1.2 has one — see
   `MEMORY.md` entry for `pytest setup gotchas`),
3. a shape we haven't tested (vLLM's production `topk` and `s_kv` are
   far larger than anything our tests cover),
4. unrelated to our integration.

## What's already wired up

Two repos changed:

### `tilelang-to-flydsl-skills/FlyvLLM/fly_vllm/kernels/sparse_attn_prefill.py`

- `rocm_ref_sparse_attn_prefill_flydsl(..., output=None)` writes into a
  caller-provided bf16 buffer when given (vLLM call style).  Falls back
  to allocate-and-copy when shape/dtype/device/contiguity don't match.
- 3-D kv normalization at the top of both wrappers
  (`if kv.dim() == 3: kv = kv.reshape(kv.shape[0], kv.shape[2])`).
- Tests in `tests/test_sparse_attn.py`:
  - `test_sparse_attn_prefill_correctness` — original 4 configs.
  - `test_sparse_attn_prefill_direct_write` — exercises the new
    `output=` path and asserts `data_ptr` identity.
  - `test_sparse_attn_prefill_vllm_call_shape` — exercises 3-D kv
    matching the vLLM call shape (regression for the first crash).

### `vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

`rocm_sparse_attn_prefill` (line 971) now:

1. Try-imports `fly_vllm.kernels.sparse_attn_prefill.rocm_ref_sparse_attn_prefill_flydsl`.
2. Logs the first call's shapes (one-shot per process — search for
   `[rocm_sparse_attn_prefill][pid=...] first call:` in the engine
   stderr / tmux output).
3. Routes through the FlyDSL fast path **unless**
   `VLLM_DISABLE_FLYDSL_SPARSE_ATTN_PREFILL=1` is set, in which case it
   falls back to the original torch reference + `output.copy_` path.
4. The original torch reference path is unchanged — set the env var
   and behaviour is identical to pre-integration.

vLLM call site (unchanged): `vllm/model_executor/layers/deepseek_v4_attention.py`
line 1026.

## To continue debugging — next steps in order

### Step 1 — confirm the segfault is from our integration

Restart vLLM with the FlyDSL path disabled:

```bash
VLLM_DISABLE_FLYDSL_SPARSE_ATTN_PREFILL=1 <vllm serve command>
```

If the crash disappears, our kernel/wrapper is the cause and we proceed
to step 2.  If it persists, the bug is elsewhere (not our problem) and
we can re-enable.

### Step 2 — capture the actual production shapes

Restart with FlyDSL re-enabled (drop the env var).  On the very first
prefill, the wrapper prints one diagnostic line per worker, e.g.:

```
[rocm_sparse_attn_prefill][pid=42906] first call:
  q=(5546, 4, 576)/torch.bfloat16/contig=True
  kv=(1097728, 1, 576)/torch.bfloat16/contig=True
  indices=(5546, 1, 6144)/torch.int32
  topk_length=(5546,)
  head_dim=512 scale=...
  attn_sink=(16,)
  output=(5546, 4, 512)/torch.bfloat16/contig=True
```

Specifically write down:

- `q.shape` and `output.shape` (do they match in the last dim?  if yes,
  `output.shape[-1] == head_dim` and the direct-write fast path fires;
  if `output.shape[-1] == d_qk` then the fallback fires, alloc+copy_
  with a shape mismatch may raise).
- `topk` (= `indices.shape[-1]`).  Standalone tests cover 32–1024;
  vLLM likely passes ≥ 4096 (`combine_topk_swa_indices` aligns to 128
  and includes `window_size`).
- `s_kv` (= `kv.shape[0]`).  Tests cover ≤ 4096; vLLM passes
  `4 * (N + window_size + max_num_batched_tokens)` which for DSv4-Flash
  is on the order of 10⁶.
- `h_q` per rank (= `n_local_heads` = 16 / TP).  Standalone tests cover
  4, 8, 16; should be fine.

### Step 3 — write a standalone reproducer

Once shapes are known, add a test like:

```python
def test_sparse_attn_prefill_vllm_actual_shape():
    s_q, s_kv, h_q, d_qk, head_dim, topk = 5546, 1097728, 4, 576, 512, 6144
    # ... allocate, run flydsl, run torch ref, compare ...
```

Run it standalone (not under pytest's `FLYDSL_RUNTIME_ENABLE_CACHE=0`
conftest) to match production cache-enabled behaviour:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=1 python -c "from tests.test_sparse_attn import \
   test_sparse_attn_prefill_vllm_actual_shape; \
   test_sparse_attn_prefill_vllm_actual_shape()"
```

If this segfaults, we have a deterministic reproducer.  If it doesn't,
the bug needs vLLM-specific state — keep reading.

### Step 4 — narrow down the segfault source

Hypotheses to check, in order of likelihood:

#### 4a.  FlyDSL 0.1.2 stale-CallState bug

Our test conftest sets `FLYDSL_RUNTIME_ENABLE_CACHE=0` to dodge it
(see `conftest.py` and the project memory note).  vLLM doesn't, so
the JIT disk cache is on.  Try launching vLLM with
`FLYDSL_RUNTIME_ENABLE_CACHE=0` (only blocks the disk cache; the
`lru_cache` on `_get_kernel` keeps in-memory caching for warm calls,
so the only cost is cold-start re-compile time).

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 <vllm serve command>
```

If this fixes it, the bug is the known FlyDSL runtime cache issue.
Make `fly_vllm` set this on import (as `tests/conftest.py` does) and
document it.

#### 4b.  topk / s_kv exceed something we silently assume

Re-read `_can_use_flydsl` and the kernel body for any 16-bit / 32-bit
overflow.  Specific things to check:

- `pos = tile_start_i32 + tid` for `tile_start_i32 = NUM_TILES *
  BLOCK_N` close to 32-bit.
- `kv_row_offset = idx * d_qk` — for `idx ~ 1M, d_qk = 576`, the byte
  offset `idx * d_qk * 2` is ~1.27 GB; near 2 GB signed-int boundary.
  `make_buffer_tensor` builds a buffer descriptor whose `numel` field
  is in bytes — verify it can represent >2 GB tensors correctly.
- `combined_topk = ceil_div(topk + window_size, 128) * 128` — much
  larger than `topk` itself.  Check `_can_use_flydsl`'s
  `topk % BLOCK_N == 0` is satisfied (BLOCK_N=32, alignment is 128, ✓).

#### 4c.  Direct-write fallback shape mismatch

If `output.shape == (s_q, h_q, d_qk=576)` (i.e. matches `q.shape` per
the assertion at `deepseek_v4_attention.py:756`) but our kernel writes
shape `(s_q, h_q, head_dim=512)`, then:

- Direct-write check fails (shape mismatch on last dim) → fallback.
- Fallback does `output.copy_(out)` where `out.shape = (s_q, h_q, 512)`
  and `output.shape = (s_q, h_q, 576)`.  PyTorch would raise on
  broadcast mismatch — *should* be a Python exception, not a segfault.
  But verify: maybe inside autograd or some custom op path it gets
  swallowed and we corrupt memory by writing only the first 512
  columns of a 576-wide buffer that downstream code reads as 576-wide.

If this is the actual issue, the fix is in the kernel wrapper:

- If `output.shape[-1] > head_dim`, accept it: only write the first
  `head_dim` columns of each `(sq, hq)` row, leave the rest alone.
  The kernel's `buffer_store` already does this naturally — we just
  need to relax the direct-write shape check from
  `output.shape == (s_q, h_q, head_dim)` to
  `output.shape[:2] == (s_q, h_q) and output.shape[-1] >= head_dim`,
  and pass the strides through.  (May need stride support in the
  kernel signature.)

#### 4d.  Memory corruption from prior ROCm kernel

Less likely but possible: aiter / triton kernel before ours leaves the
GPU in a weird state, our kernel exposes it.  Check by swapping order:
launch with FlyDSL disabled (env var), confirm no crash; then enable.
If only enabled crashes but order is the same, it's our kernel.

### Step 5 — once root cause is known

- If the fix is in the kernel wrapper (`sparse_attn_prefill.py`), add a
  regression test mirroring the production shape and tighten
  `_can_use_flydsl`.
- If the fix is in vLLM (`rocm_aiter_mla_sparse.py`), keep the env-var
  escape hatch for safety.
- Remove the one-shot diagnostic logging once the bug is well
  understood (it currently prints once per worker on first call).
- Update `sparse_attn_prefill_PERF.md` with the production shape's
  measured perf so future tuning has the right baseline.

## Files to touch (and not touch)

- **Kernel and wrappers**:
  `tilelang-to-flydsl-skills/FlyvLLM/fly_vllm/kernels/sparse_attn_prefill.py`
- **Standalone tests**:
  `tilelang-to-flydsl-skills/FlyvLLM/tests/test_sparse_attn.py`
- **vLLM integration point** (only the `rocm_sparse_attn_prefill`
  function, ~30 lines):
  `vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

The `vllm` repo's call site at
`vllm/model_executor/layers/deepseek_v4_attention.py:1026` is
*unchanged*; per `vllm/AGENTS.md` we don't make broader vLLM edits
without an explicit owner-justification.

## Reference: what the integration is supposed to save

For DSv4 production shapes (estimated from standalone benches):

| config         | torch ref + copy_ | flydsl direct-write | speedup |
|----------------|-------------------|---------------------|---------|
| v4flash_tp4    | ~289 us           | ~136 us             | 2.13×   |
| v4pro_tp8      | ~287 us           | ~248 us             | 1.16×   |

If you decide the integration isn't worth the debugging complexity, the
clean revert is:

1. Drop the import-and-dispatch block in
   `rocm_aiter_mla_sparse.py:954–1015` (keep only the original
   `rocm_sparse_attn_prefill` body).
2. Leave the kernel-side changes in `sparse_attn_prefill.py` —
   the `output=` param is backward-compatible and the 3-D kv guard is
   defensive cleanup either way.
