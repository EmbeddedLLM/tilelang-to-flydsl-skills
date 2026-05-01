---
name: TileKernels port project context
description: Background on the FlyTileKernels port project — gfx950 target, what's installed, what was already done, what was left to do.
type: project
originSessionId: c032df3a-5434-4761-a4fd-491e1977a619
---
The user (tunjian.tan@embeddedllm.com) is porting the ~36-kernel
DeepSeek `TileKernels` package (`/app/flytilelang/TileKernels`) from
TileLang to FlyDSL.  Target hardware: AMD gfx950 (CDNA 4 / MI350 /
MI355X).  TileLang doesn't run on ROCm so the port is mandatory.

**Why:** The port unlocks vLLM integration (per `CLAUDE.md`).  The user
also needs CUDA→ROCm fp8 weight-loader helpers but those are deferred.

**How to apply:** When asked about TileKernels/FlyTileKernels work,
remember:

- Repo layout: skill is at
  `/app/flytilelang/tilelang-to-flydsl-skills/.claude/skills/tilelang-to-flydsl/`,
  ports at `/app/flytilelang/tilelang-to-flydsl-skills/FlyTileKernels/`.
- FlyDSL is pip-installed (v0.1.2) at
  `/usr/local/lib/python3.12/dist-packages/flydsl`.  No working FlyDSL
  example kernels are installed — patterns must be discovered by
  reading the source and trial-and-error.  See the `flydsl_api_quirks`
  memory for substitutes that work.
- gfx950 specifics: wave size 64, 160 KB LDS/CU, OCP fp8 variants
  (`Float8E4M3FN`, `Float8E5M2`) — NOT the FNUZ variants used by gfx94x.
  Helpers in `_flydsl_helpers.py` are already wired for gfx950.
- The skill's `STATUS.md` listed cross-cutting blockers (integer LDS
  atomics, integer BufferCopy).  Integer BufferCopy turned out to NOT be
  a blocker — `BufferCopy64b()` works fine for `Int64`.  Only
  integer-LDS-atomic / integer-global-atomic kernels (`group_count`,
  `aux_fi`, `get_fused_mapping`) actually need a workaround.
- User direction (April 2026): "first validation batch only" — toolchain
  + a few simple kernels.  Test each before moving on.  Workaround
  blockers in user code rather than patching FlyDSL.

## What got done in the first validation batch (2026-04-30)

Three kernels green: `moe.normalize_weight` (12 tests), `mhc.expand` (36
tests, fwd+bwd+autograd), `moe.mask_indices_by_tp` (36 tests). Total 84
passing.  Working patterns documented inline and in
`STATUS.md` "FlyDSL API quirks" section.

## What got done in the second batch (2026-04-30 cont.)

Two more kernels green: `engram.engram_hash` (2 tests), `transpose`/
`batched_transpose` for bf16 + fp32 (70 tests).  Total 156 passing
across 5 kernels.

`quant.cast_back` was attempted but parked: blocked on the fp8 → fp32
cast lowering — `arith.extf f8E4M3FN → f32` produces an
`unrealized_conversion_cast` that the FlyDSL→LLVM pipeline cannot
resolve for OCP fp8 on gfx950, regardless of whether you use
`Float32(fp8_val)` or call `arith.extf` directly.  `STATUS.md` has the
detailed write-up.

`transpose` for fp8 (e4m3) was also parked: 42 tests fail with
NotImplementedError because BufferCopy minimum is 16 bits, so single-byte
transposed stores would race between adjacent threads.  Needs LDS
pair-and-pack to land.

## What's next

The fp8 conversion path is the next significant unlock.  See `STATUS.md`
"Recommended order for completing the port" — three candidate strategies
listed (`rocdl.cvt.f32.fp8` direct, the `rocdl.cvt.scalef32.pk.f32.fp8`
family for cast_back specifically, or manual bit-twiddling).  After
that, the LDS pair-and-pack for fp8 transpose, then the `quant/` family
opens up.
