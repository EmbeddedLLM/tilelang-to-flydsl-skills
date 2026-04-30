# FlyTileKernels

A FlyDSL re-implementation of the
[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels)
operator collection, targeting AMD ROCm and the **gfx950** architecture
(MI350 / MI355X).

The goal is to keep TileKernels' public API and pytest suite intact while
swapping the underlying kernel DSL from TileLang (TVM TIR-based, currently
unsupported on ROCm) to
[FlyDSL](https://github.com/ROCm/FlyDSL) (MLIR-native, ROCm-first).

## Status

**Work in progress.**  See [STATUS.md](STATUS.md) for the per-kernel state.

In short:

- The package skeleton, helpers, testing utilities, torch reference impls,
  and the entire pytest suite are in place with `tile_kernels` →
  `fly_tile_kernels` import rewrites.
- A small set of kernels is fully ported (currently:
  `moe.normalize_weight`).
- The remaining ~30 kernels raise `NotImplementedError` with a one-line
  reason and a docstring sketching the FlyDSL pattern needed to complete
  the port.
- A few cross-cutting blockers in FlyDSL (integer LDS / global atomics,
  integer BufferCopy load/store skeleton) would unlock most of the
  remaining stubs in one batch.  See STATUS.md.

This snapshot was produced *without* running the code.  Validation requires
a gfx950 machine with FlyDSL installed.

## Layout

```
fly_tile_kernels/
├── __init__.py
├── _flydsl_helpers.py        # shared dtype mapping, copy-atom selection,
│                             # wave/block reduction butterfly
├── _stub.py                  # standardised "not yet ported" helper
├── config.py                 # SM accounting (mirrors tile_kernels.config)
├── utils.py                  # ceil_div / align / is_power_of_two
├── moe/                      # MoE routing, scoring, top-k, expansion
├── quant/                    # FP8 / FP4 / E5M6 cast and dequant
├── transpose/                # Batched 2-D transpose
├── mhc/                      # Manifold HyperConnection kernels
├── engram/                   # Engram gating kernels
├── modeling/                 # torch.autograd.Function wrappers
├── testing/                  # Test/benchmark helpers (copied verbatim)
└── torch/                    # PyTorch reference implementations (copied verbatim)

tests/                        # Upstream pytest suite, imports rewritten
```

## Installing

```sh
# Prerequisite: a working FlyDSL build (see FlyDSL/scripts/build*.sh).
pip install -e ".[dev]"
```

## Running tests on a gfx950 machine

```sh
# The currently-ported kernel:
pytest tests/moe/test_normalize_weight.py -v

# Full suite (most tests will fail with NotImplementedError + STATUS.md
# pointer until the remaining kernels are ported):
pytest tests/ -n 4
```

## Hardware target details

| Property | Value |
|---|---|
| Architecture | `gfx950` (CDNA 4 / MI350 / MI355X) |
| Wave size | 64 |
| LDS / CU | 160 KB |
| FP8 variants | OCP `Float8E4M3FN`, `Float8E5M2` (not the FNUZ variants used by gfx94x) |
| BF16 → FP32 | Hardware pack instruction available |

If/when porting to other arches:

- gfx94x (MI300X / MI308X): switch fp8 to FNUZ variants.
- RDNA / gfx1250: wave size becomes 32; reductions in `_flydsl_helpers.py`
  read `get_warp_size()` automatically.

## Contributing

The conversion approach is documented in the
[tilelang-to-flydsl skill](../.claude/skills/tilelang-to-flydsl/SKILL.md)
that lives in this same parent repo.  In particular:

- `references/api_mapping.md` — symbol-by-symbol translation table.
- `references/idioms.md` — side-by-side patterns for the five kernel
  archetypes.
- `references/gotchas.md` — review checklist; *read before declaring a
  port done*.
- `references/workflow.md` — per-kernel procedure.

When converting a stubbed kernel, follow the workflow and update the
relevant row of `STATUS.md`.

## License

MIT (matches upstream TileKernels).
