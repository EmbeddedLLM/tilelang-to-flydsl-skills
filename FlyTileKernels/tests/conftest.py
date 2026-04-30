# Root-level conftest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.

import os

# Disable FlyDSL's runtime kernel cache during testing.  Some kernels (e.g.
# `normalize_weight`) hit a `CallState` reuse bug when two tests share the
# same launcher signature but different tensor shapes — the second test
# observes stale outputs from the first.  Disabling the cache costs a
# small amount of compile time per test but produces reliable results.
os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '0')

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]
