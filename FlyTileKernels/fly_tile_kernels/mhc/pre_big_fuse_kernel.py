"""mhc_pre_big_fuse: composite of norm_fn + split_mixes + sinkhorn +
pre_apply_mix.  The test asserts ``torch.equal`` against the unfused
reference (``mhc_pre_norm_fn`` → ``mhc_pre_split_mixes`` →
``sinkhorn_normalize`` → ``mhc_pre_apply_mix``), so this implementation
calls the same kernels rather than reimplementing the math.

For sinkhorn we route through the FlyDSL kernel via
``_mhc_sinkhorn_fwd`` to match the unfused reference bit-exactly.
"""

import torch

from fly_tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_fwd


def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
):
    K = mhc_mult * hidden_size
    M2 = mhc_mult * mhc_mult
    M3 = mhc_mult * 2 + M2

    def runner(
        gemm_out_mul,        # (n_splits, n, M3) fp32 — from fwd_mul
        gemm_out_sqrsum,     # (n_splits, n) fp32     — from fwd_mul
        mhc_scale,           # (3,) fp32
        mhc_base,            # (M3,) fp32
        residual_flat,       # (n, mhc_mult, hidden_size) bf16
        post_mix,            # (n, mhc_mult)              fp32 out
        comb_mix,            # (n, mhc_mult^2)            fp32 out
        layer_input,         # (n, hidden_size)           bf16 out
    ):
        # 1. Reduce splits, then apply RMS-norm rsqrt → mixes  (matches
        #    ``_mhc_pre_norm_fn_fwd_norm``).
        out_mul = gemm_out_mul.sum(0)        # (n, M3)
        sqrsum = gemm_out_sqrsum.sum(0)      # (n,)
        # The fwd_norm in norm_fn_kernel keeps a singleton dim for the
        # reduction — reproduce it here for bit-exact parity.
        out_mul_3d = out_mul.unsqueeze(1)    # (n, 1, M3)
        sqrsum_2d = sqrsum.unsqueeze(-1)     # (n, 1)
        rsqrt = (sqrsum_2d.unsqueeze(-1) / K + rms_eps).rsqrt()  # (n, 1, 1)
        mixes = (out_mul_3d * rsqrt).sum(-2)  # (n, M3)

        # 2. Split-mixes (matches ``_mhc_pre_split_mixes_fwd``).
        m = mhc_mult
        pre_layer_mix = torch.sigmoid(mixes[:, :m] * mhc_scale[0] + mhc_base[:m]) + mhc_pre_eps
        post_layer_mix = torch.sigmoid(
            mixes[:, m:2 * m] * mhc_scale[1] + mhc_base[m:2 * m],
        ) * mhc_post_mult_value
        comb_res_mix = mixes[:, 2 * m:] * mhc_scale[2] + mhc_base[2 * m:]
        post_mix.copy_(post_layer_mix)

        # 3. Sinkhorn-normalise comb (matches ``sinkhorn_normalize``).
        comb_in = comb_res_mix.view(-1, m, m).contiguous()
        comb_out = torch.empty_like(comb_in)
        sinkhorn_runner = _mhc_sinkhorn_fwd(
            m, 1, sinkhorn_repeat, mhc_sinkhorn_eps,
        )
        sinkhorn_runner(comb_in, comb_out)
        comb_mix.copy_(comb_out.view(-1, M2))

        # 4. pre_apply_mix:  layer_input = sum_m pre[m] * residual[m]  (bf16).
        # pre_layer_mix is shape (n, m); residual_flat is (n, m, h).
        pre_4d = pre_layer_mix.unsqueeze(-1).float()         # (n, m, 1)
        contrib = (residual_flat.float() * pre_4d).sum(-2)   # (n, h)
        layer_input.copy_(contrib.bfloat16())
    return runner
