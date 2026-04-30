# tilelang-to-flydsl-skills

A Claude Code skill that teaches an agent how to port kernels written in
**TileLang** (the `@T.prim_func` Python DSL used by
[tile-ai/tilelang](https://github.com/tile-ai/tilelang) and
[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels))
into equivalent **FlyDSL** kernels (the Python DSL on top of the
`fly` MLIR dialect, targeting AMD ROCm).

The skill is review-driven: it assumes the agent may not be able to build or
run either project, and uses a structured workflow + checklist to validate
ports without execution. Final correctness is gated on the user running the
target project's pytest suite.

## Layout

```
.claude/skills/tilelang-to-flydsl/
├── SKILL.md                                  # Skill entry-point (frontmatter + workflow)
├── README.md                                 # Skill-level overview
└── references/
    ├── api_mapping.md                        # T.* → fx.* symbol table
    ├── idioms.md                             # Side-by-side patterns for the 5 archetypes
    ├── gotchas.md                            # Review checklist (read before declaring done)
    ├── workflow.md                           # Step-by-step procedure per kernel
    └── worked_examples/
        ├── normalize_weight.md               # Small reduction kernel — full conversion
        ├── batched_transpose.md              # Shared-memory rearrange + strided input
        └── gemm_skeleton.md                  # Annotated MFMA GEMM skeleton
```

## Using the skill in another project

Drop the skill into the target repo's `.claude/skills/` directory:

```sh
git clone https://github.com/<your-org>/tilelang-to-flydsl-skills.git /tmp/tk2fly
mkdir -p <your-repo>/.claude/skills
cp -r /tmp/tk2fly/.claude/skills/tilelang-to-flydsl <your-repo>/.claude/skills/
```

Or, if the target repo is itself a Claude Code project, add this repo as a
git submodule under `.claude/skills/` (less common, but works).

Once the directory is in place, Claude Code will list the skill among
available skills inside that repo and trigger it on prompts like
"port this TileLang kernel to FlyDSL", "convert tile_kernels/.../X to
FlyDSL", or "rewrite this `@T.prim_func` for FlyDSL".

## What the skill covers

- The two-DSL mental model and the four hard sub-problems (launch shape,
  memory map, loop nest, data movement).
- A complete TileLang → FlyDSL symbol mapping table (decorators, types,
  allocations, loops, sync, copies, GEMM, reductions, atomics, math
  intrinsics, layout hints, debugging).
- Side-by-side idioms for the five common kernel archetypes:
  vectorised elementwise/cast, per-thread distributed loop (`T.Parallel`),
  shared-memory rearrange (transpose), reduction (norm / softmax / topk),
  GEMM via MFMA atoms.
- A 27-item gotcha review checklist, with the silent-correctness bugs
  highlighted.
- A per-kernel 8-step workflow that frames the port as research → sketch →
  fill → audit → diff → report.
- Three full worked examples spanning small (one thread per row),
  medium (LDS + strided input + custom thread layout), and large (MFMA
  GEMM skeleton).

## What the skill does not cover

- **Performance tuning.** A faithful port is the goal; performance work is
  out of scope. Hand off to the FlyDSL-side skills (`gemm-optimization`,
  `lds-optimization`, `prefetch-data-load`, `kernel-trace-analysis`).
- **Toolchain build/install.** Use FlyDSL's `build-flydsl` skill.
- **Debugging a wrong port.** Use FlyDSL's `debug-flydsl-kernel` skill
  after this one's review pass.

## License

MIT (or whatever the parent project chooses — fill in before publishing).
