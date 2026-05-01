---
name: pytest setup gotchas for FlyTileKernels
description: Steps required to make pytest work on a fresh checkout of FlyTileKernels — install workaround, sed fix, cache disable.
type: reference
originSessionId: c032df3a-5434-4761-a4fd-491e1977a619
---
To run any pytest in `/app/flytilelang/tilelang-to-flydsl-skills/FlyTileKernels`,
three issues must be fixed first.  These are persistent properties of the
repo as it stands (April 2026), not transient environment state.

**Why:** Without these the tests don't even *collect*; the user
otherwise hits cryptic ImportError chains.

**How to apply:** Use this checklist when re-onboarding a session that
intends to run the test suite.

## 1. pip install with a pretend version

`pyproject.toml` uses `setuptools-scm` (via the `vcs-versioning` fork)
which fails because `git introspection` walks up into the parent skills
git repo and trips a "dubious ownership" check.

Workaround:

```sh
GIT_CEILING_DIRECTORIES=/app/flytilelang/tilelang-to-flydsl-skills \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLY_TILE_KERNELS=0.0.1 \
  pip install -e ".[dev]"
```

Both env vars are needed — `GIT_CEILING_DIRECTORIES` stops the git walk,
`SETUPTOOLS_SCM_PRETEND_VERSION_*` provides a version since git lookup
fails.

## 2. Fix the `fly_fly_tile_kernels` typo across tests

39 test files contain `fly_fly_tile_kernels` instead of `fly_tile_kernels`
— a leftover from a bad bulk sed during the upstream port.  Fix once:

```sh
grep -rl 'fly_fly_tile_kernels' tests/ | xargs sed -i 's/fly_fly_tile_kernels/fly_tile_kernels/g'
```

Verify empty: `grep -rn 'fly_fly_tile_kernels' tests/`.

## 3. Disable the FlyDSL runtime cache during tests

`tests/conftest.py` already sets `FLYDSL_RUNTIME_ENABLE_CACHE=0`.  This
works around a `CallState` reuse bug where two tests with the same
kernel signature but different tensor data observe stale outputs.  Don't
remove it without re-reproducing the bug.

## Currently-green tests on gfx950

```
pytest tests/moe/test_normalize_weight.py \
       tests/moe/test_mask_indices_by_tp.py \
       tests/mhc/test_expand.py
```

84 passed, 48 skipped (benchmarks).
