"""FlyTileKernels: TileKernels reimplemented on FlyDSL for AMD ROCm (gfx950).

Drop-in compatible with the deepseek-ai/TileKernels public API.  All public
kernel wrappers preserve their original signatures so the upstream pytest
suite can be reused with only an import-path rewrite.
"""

from . import (
    config,
    engram,
    mhc,
    modeling,
    moe,
    quant,
    transpose,
    torch,
    testing,
    utils,
)

from .config import get_num_sms, get_device_num_sms, set_num_sms
