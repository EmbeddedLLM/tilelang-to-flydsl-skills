"""Cast-config helpers shared by the quant kernels.

The TileKernels original carries a few `@T.macro` helpers (`get_sf_and_inv`,
`load_sf`, `transform_sf`, `store_sf`) here that are inlined into
`@T.prim_func` bodies.  FlyDSL has no macro mechanism, so the *kernel-side*
implementations are inlined directly into each ported kernel.  This module
keeps only the *host-side* dataclass / shape / allocation helpers, plus
`unpack_from_e2m1fn_x2` for fp4 debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

import torch

from fly_tile_kernels.quant.types import QuantTensor
from fly_tile_kernels.utils import align, ceil_div


# ---------------------------------------------------------------------------
# Vectorisation budget.  TileKernels picked this from the NVIDIA target
# compute capability.  On gfx950 the equivalent question is "how many bytes
# can a single AMD `buffer_load` move?" -- 16 bytes (BufferCopy128b).
# ---------------------------------------------------------------------------

def get_best_vectorize_size(torch_dtype: torch.dtype) -> int:
    """Largest vector width (in elements) that still fits a 128-bit copy.

    Mirrors the upstream helper but always picks the gfx950 16-byte budget.
    """
    if torch_dtype in (torch.float32, torch.int32, torch.uint32):
        return 4
    if torch_dtype in (torch.float16, torch.bfloat16, torch.int16, torch.uint16):
        return 8
    if torch_dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
        return 16
    if torch_dtype == torch.float64 or torch_dtype == torch.int64:
        return 2
    return 16


# ---------------------------------------------------------------------------
# Cast configuration dataclasses.  Same shape as the TileKernels originals,
# but the `.dtype` property returns a *string* identifier rather than a
# `T.dtype`, since FlyDSL kernels read the dtype as a Constexpr[str].
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False

    @property
    def dtype_str(self) -> str:
        if self.torch_dtype == torch.int8:
            # int8 is a packed fp4 (e2m1) container in TileKernels.
            return "fp4_e2m1"
        return str(self.torch_dtype).replace("torch.", "")

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype_str(self) -> str:
        return "uint8" if self.use_packed_ue8m0 else "float32"


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    with_sf: bool = True


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    round_sf: bool = False
    custom_clamp_min_value: Optional[float] = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        if self.torch_dtype == torch.float8_e4m3fn:
            return 1e-4
        if self.torch_dtype == torch.int8:
            # fp4 e2m1: max 6.0, min subnormal 0.5 * 2^(1-1) = 0.5;
            # the original uses max_value * 2^-126.
            return 6.0 * (2 ** -126)
        raise ValueError(f"unsupported dtype {self.torch_dtype}")


def get_cast_input_and_config(
    x: Union[torch.Tensor, QuantTensor],
    sf_block: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, Optional[torch.Tensor], CastInputConfig]:
    if isinstance(x, tuple):
        assert isinstance(sf_block, tuple)
        x, x_sf = x
        config = CastInputConfig(torch_dtype=x.dtype, with_sf=True, sf_block=sf_block)
        assert isinstance(x, torch.Tensor) and isinstance(x_sf, torch.Tensor)
        assert x.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        if x_sf.stride(0) == 1:
            config = replace(config, use_tma_aligned_col_major_sf=True)
            x_sf = x_sf.T
            if x_sf.dtype == torch.int32:
                config = replace(config, use_packed_ue8m0=True)
                x_sf = x_sf.view(torch.uint8)
        else:
            assert x_sf.stride(1) == 1
            assert x_sf.dtype == torch.float32
        return x, x_sf, config
    config = CastInputConfig(torch_dtype=x.dtype, with_sf=False)
    assert sf_block is None
    assert isinstance(x, torch.Tensor)
    assert x.dtype in (torch.bfloat16, torch.float32)
    return x, None, config


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: Optional[float] = None,
) -> CastOutputConfig:
    assert fmt in ("e5m6", "e4m3", "e2m1")
    mapping = {
        "e5m6": torch.uint32,
        "e4m3": torch.float8_e4m3fn,
        "e2m1": torch.int8,
    }
    return CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )


def get_logical_hidden(hidden: int, dtype: torch.dtype) -> int:
    return hidden if dtype != torch.int8 else hidden * 2


def get_physical_hidden(hidden: int, dtype: torch.dtype) -> int:
    return hidden if dtype != torch.int8 else hidden // 2


def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_block_m = num_block_m * 4
        num_block_k = ceil_div(num_block_k, 4)
    return (num_block_k, num_block_m) if config.use_tma_aligned_col_major_sf else (num_block_m, num_block_k)


def alloc_scaling_factors(shape, out_config: BaseCastConfig, device="cuda") -> torch.Tensor:
    sf_shape = get_sf_shape(shape, out_config)
    aligned_sf_shape = sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        aligned_sf_shape = align(sf_shape[1], 16 if out_config.use_packed_ue8m0 else 4)
    sf = torch.empty(size=(sf_shape[0], aligned_sf_shape), dtype=out_config.sf_torch_dtype, device=device)
    if out_config.use_tma_aligned_col_major_sf:
        sf = sf[:, : sf_shape[1]]
    return sf


def cast_epilogue(out_sf: torch.Tensor, num_tokens: int, hidden: int, config: BaseCastConfig) -> torch.Tensor:
    if config.use_packed_ue8m0:
        if num_tokens == 0:
            out_sf = torch.empty((out_sf.shape[0], out_sf.shape[1] // 4), dtype=torch.int32, device=out_sf.device)
        else:
            out_sf = out_sf.view(dtype=torch.int32)
    out_sf = out_sf.T if config.use_tma_aligned_col_major_sf else out_sf
    out_sf = out_sf[: ceil_div(num_tokens, config.sf_block[0]), :]
    return out_sf


# ---------------------------------------------------------------------------
# fp4 (e2m1) unpack helper -- pure torch, lifted verbatim from upstream.
# ---------------------------------------------------------------------------

def unpack_from_e2m1fn_x2(x: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode a uint8/int8 tensor of packed fp4 values to float32 (debug only)."""
    assert x.dtype == torch.int8 or x.dtype == torch.uint8
    if x.ndim == 0:
        raise ValueError("x must have at least 1 dimension so the last dim can be doubled")
    lo = (x & 0x0F).to(torch.int16)
    hi = ((x >> 4) & 0x0F).to(torch.int16)

    def decode_fp4_e2m1(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1
        sign = torch.where(s == 1,
                           torch.tensor(-1.0, device=n.device),
                           torch.tensor(1.0, device=n.device))
        bias = 1
        sub = (m.to(torch.float32) * 0.5) * (2.0 ** (1 - bias))
        norm = (1.0 + m.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device), (e - bias).to(torch.float32))
        val = torch.where(e == 0, sub, norm)
        return (val * sign).to(out_dtype)

    flo = decode_fp4_e2m1(lo)
    fhi = decode_fp4_e2m1(hi)
    y = torch.stack([flo, fhi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)
    return y
