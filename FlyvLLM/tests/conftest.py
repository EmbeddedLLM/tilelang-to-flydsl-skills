"""Disable the FlyDSL runtime kernel cache during tests.

The runtime cache has a stale-CallState reuse bug where two test cases
sharing a kernel signature observe each other's data.  Setting this env
var before any flydsl import works around it.  Mirrors the pattern used
by FlyTileKernels.
"""

import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
