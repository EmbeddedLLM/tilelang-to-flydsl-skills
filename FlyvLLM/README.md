# FlyvLLM

FlyDSL kernels for the vLLM MHC layer, targeted at AMD gfx950 (CDNA4 / MI355X).

A drop-in replacement for `vllm.model_executor.layers.mhc` on ROCm where
the upstream TileLang kernels do not run.

## Public API

```python
from fly_vllm.mhc import mhc_pre, mhc_post

post_mix, comb_mix, layer_input = mhc_pre(
    residual, fn, hc_scale, hc_base,
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
    sinkhorn_repeat,
)
out = mhc_post(x, residual, post_layer_mix, comb_res_mix)
```

Both functions are inference-only (no autograd) and match the tensor
shapes/dtypes of the upstream vLLM implementation.

## Layout

```
fly_vllm/
├── mhc.py                          # Public mhc_pre / mhc_post wrappers
└── kernels/
    ├── tf32_hc_prenorm_gemm.py     # DeepGEMM replacement (torch.matmul)
    ├── mhc_norm_split_kernel.py    # Reduce splits + RMS rsqrt + sigmoid splits
    ├── sinkhorn_kernel.py          # Iterative row/col normalisation (mhc x mhc)
    ├── pre_apply_mix_kernel.py     # residual * pre_mix → layer_input (bf16)
    └── mhc_post_kernel.py          # Single fused output kernel
tests/
├── test_mhc_pre.py
└── test_mhc_post.py
```

## How `mhc_pre` is decomposed

The upstream TileLang `mhc_pre_big_fuse` deeply fuses several operations
into one kernel that splits work between threads `< 32` (sinkhorn + post +
comb) and threads `>= 32` (pre + apply_mix).  Porting that exact thread
split to FlyDSL is non-trivial in this build; the FlyvLLM port instead
chains four FlyDSL kernels:

1. `tf32_hc_prenorm_gemm` — split-K bf16⇒fp32 matmul + per-row sqrsum.
   On ROCm we route to `torch.matmul` with `allow_tf32=True`; hipBLASLt
   is the right backend at these sizes (collapses the split into a single
   call and parks the result in slot 0 of the partial buffer).
2. `mhc_norm_split_kernel` — sum splits, compute RMS rsqrt, multiply
   mixes by rsqrt, then split into pre/post/comb_unnormalised.
3. `sinkhorn_kernel` — iterative row/col normalisation of the
   per-token `(mhc, mhc)` matrix; one thread per token with full state in
   registers.
4. `pre_apply_mix_kernel` — weighted sum over the mhc dim, bf16 in/out.

## How `mhc_post` maps

A single fused FlyDSL kernel mirroring the TileLang one: one thread per
`(token, h)` element, mhc unrolled in registers.

## Testing

```sh
cd /app/flytilelang/tilelang-to-flydsl-skills/FlyvLLM
pip install -e ".[dev]"
pytest tests/
```

`tests/conftest.py` sets `FLYDSL_RUNTIME_ENABLE_CACHE=0` to work around
the FlyDSL 0.1.2 stale-`CallState` bug (same fix as FlyTileKernels uses).
