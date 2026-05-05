"""Tiny benchmarking helper used by the rocm_aiter_mla_sparse tests.

Wraps `torch.cuda.Event`-based timing.  Each variant is run for a few warm-up
iterations, then for ``iters`` timed iterations; the median is returned.

The benchmark helper temporarily enables the FlyDSL runtime kernel cache so
the per-call overhead is comparable to a deployed setup.  In tests we
disable the cache to dodge a CallState reuse bug between unrelated tests;
benchmarks call into the cache deliberately for a fair apples-to-apples
measurement.
"""

from __future__ import annotations

import contextlib
import os
import statistics
from typing import Callable

import torch


@contextlib.contextmanager
def _flydsl_cache_enabled():
    prev = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE")
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("FLYDSL_RUNTIME_ENABLE_CACHE", None)
        else:
            os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = prev


def bench(fn: Callable[[], object], *, iters: int = 25, warmup: int = 5) -> float:
    """Median wall-clock time per call, in microseconds."""
    with _flydsl_cache_enabled():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def report(label: str, results: dict[str, float]) -> str:
    """Format a one-line benchmark report for printing in test output."""
    fastest = min(results.values())
    parts = [
        f"{name}={t:8.2f}us ({fastest / t * 100:5.1f}%)" for name, t in results.items()
    ]
    return f"[{label}] " + "  ".join(parts)
