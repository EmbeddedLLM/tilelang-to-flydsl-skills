"""Standardised "not yet ported" stub helpers.

A kernel module that has not been fully translated to FlyDSL exposes its
public function as a wrapper around `not_yet_ported`.  This way:

- The module imports cleanly — no missing-attribute errors at import time.
- The pytest collection step succeeds.
- Tests that exercise the unported kernel fail with a single clear message
  rather than an opaque trace from inside FlyDSL.
- STATUS.md can list the exact set of unported items by grepping for this
  helper.
"""

from __future__ import annotations


def not_yet_ported(name: str, reason: str = ""):
    """Raise NotImplementedError with a uniform message.

    Args:
        name:   public function or kernel name.
        reason: optional one-line description of what's blocking the port
                (typically a FlyDSL primitive that hasn't been verified or
                a TileLang feature that has no direct counterpart).
    """
    msg = (
        f"FlyTileKernels: `{name}` has not been fully ported from TileLang to "
        "FlyDSL yet. See STATUS.md for the current state and the list of "
        "FlyDSL primitives needed to complete the port."
    )
    if reason:
        msg += f"\nBlocking issue: {reason}"
    raise NotImplementedError(msg)
