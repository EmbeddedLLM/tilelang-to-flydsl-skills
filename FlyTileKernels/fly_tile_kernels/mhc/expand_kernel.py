"""expand_kernel: stubbed."""

from fly_tile_kernels._stub import not_yet_ported


def __getattr__(name):
    def _stub(*args, **kwargs):
        not_yet_ported(f"mhc.expand_kernel.{name}", "MHC kernel not yet ported")
    return _stub
