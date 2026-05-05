"""mhc_norm_split: combined RMS-rsqrt step + pre/post/comb-mix split.

This kernel implements the prefix of ``mhc_pre_big_fuse_tilelang`` up to
sinkhorn (i.e. everything before the comb-only sinkhorn iterations and
before ``pre_apply_mix``).  Math, per token ``t``:

    sqsum_total = sum_split gemm_out_sqrsum[split, t]
    rsqrt       = (sqsum_total / (mhc * hidden) + rms_eps) ** -0.5

    mixes[j]    = sum_split gemm_out_mul[split, t, j]
    mixes[j]    = mixes[j] * rsqrt                      (j in [0, M3))

    pre [t, j]  = sigmoid(mixes[j]                * scale[0] + base[j])             + pre_eps   (j in [0, mhc))
    post[t, j]  = sigmoid(mixes[j+mhc]            * scale[1] + base[j+mhc])         * post_mult (j in [0, mhc))
    comb[t, j, k] = mixes[j*mhc + k + 2*mhc]      * scale[2] + base[j*mhc+k+2*mhc]              (j, k in [0, mhc))

Layout: one thread per token, all M3 mixes computed inline (constexpr
unrolled); ``mhc=4`` ⇒ M3=24 is small enough to keep in registers.

Outputs are written to three separate tensors:
- ``pre_mix``: (n, mhc) fp32
- ``post_mix``: (n, mhc) fp32
- ``comb_unnormed``: (n, mhc, mhc) fp32  (passed to sinkhorn next)
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, math as fxmath
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def _build_norm_split_kernel(
    mhc_mult: int,
    hidden_size: int,
    n_splits: int,
    rms_eps: float,
    pre_eps: float,
    post_mult: float,
):
    M = mhc_mult
    M3 = 2 * M + M * M
    K = M * hidden_size

    @flyc.kernel
    def kernel(
        gemm_out_mul:    fx.Tensor,   # (n_splits, n, M3) fp32
        gemm_out_sqrsum: fx.Tensor,   # (n_splits, n)     fp32
        hc_scale:        fx.Tensor,   # (3,)              fp32
        hc_base:         fx.Tensor,   # (M3,)             fp32
        pre_mix:         fx.Tensor,   # (n, M)            fp32 out
        post_mix:        fx.Tensor,   # (n, M)            fp32 out
        comb_unnormed:   fx.Tensor,   # (n, M, M)         fp32 out
        num_tokens:      fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        gm_buf = fx.rocdl.make_buffer_tensor(gemm_out_mul)
        gs_buf = fx.rocdl.make_buffer_tensor(gemm_out_sqrsum)
        sc_buf = fx.rocdl.make_buffer_tensor(hc_scale)
        bs_buf = fx.rocdl.make_buffer_tensor(hc_base)
        pre_buf = fx.rocdl.make_buffer_tensor(pre_mix)
        post_buf = fx.rocdl.make_buffer_tensor(post_mix)
        cb_buf = fx.rocdl.make_buffer_tensor(comb_unnormed)

        scalar_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1 = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            # ---- RMS rsqrt: sum sqsum across splits, then rsqrt(.../K + eps).
            sq_total = fx.Float32(0.0)
            for s in range_constexpr(n_splits):
                col = fx.slice(gs_buf, (s, None))
                col_div = fx.logical_divide(col, fx.make_layout(1, 1))
                r = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(col_div, (None, token)), r)
                v = fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                sq_total = sq_total + v
            rsqrt_val = fx.Float32(
                fxmath.rsqrt(sq_total / fx.Float32(K) + fx.Float32(rms_eps))
            )

            # ---- Load scale[0..2] and base[0..M3-1] once per thread.
            sc_div = fx.logical_divide(sc_buf, fx.make_layout(1, 1))
            bs_div = fx.logical_divide(bs_buf, fx.make_layout(1, 1))
            sc_vals = []
            for k in range_constexpr(3):
                r = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(sc_div, (None, k)), r)
                sc_vals.append(
                    fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                )
            sc0, sc1, sc2 = sc_vals
            base_vals = []
            for j in range_constexpr(M3):
                r = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(bs_div, (None, j)), r)
                base_vals.append(
                    fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                )

            # ---- mixes[j] = (sum_split gemm_out_mul[s, t, j]) * rsqrt
            mixes = []
            for j in range_constexpr(M3):
                tot = fx.Float32(0.0)
                for s in range_constexpr(n_splits):
                    plane = fx.slice(gm_buf, (s, token, None))
                    plane_div = fx.logical_divide(plane, fx.make_layout(1, 1))
                    r = fx.memref_alloca(scalar_ty, s_lay)
                    fx.copy_atom_call(copy_atom_32, fx.slice(plane_div, (None, j)), r)
                    v = fx.Float32(
                        fxvec.extract(fx.memref_load_vec(r), static_position=[0])
                    )
                    tot = tot + v
                mixes.append(tot * rsqrt_val)

            # ---- Write pre[j] = sigmoid(mixes[j] * sc0 + base[j]) + pre_eps
            pre_row = fx.slice(pre_buf, (token, None))
            pre_row_div = fx.logical_divide(pre_row, fx.make_layout(1, 1))
            for j in range_constexpr(M):
                z = mixes[j] * sc0 + base_vals[j]
                neg_z = fx.Float32(0.0) - z
                e = fx.Float32(fxmath.exp(neg_z))
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + e)
                ov = sig + fx.Float32(pre_eps)
                outr = fx.memref_alloca(scalar_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [ov]), outr)
                fx.copy_atom_call(copy_atom_32, outr, fx.slice(pre_row_div, (None, j)))

            # ---- Write post[j] = sigmoid(mixes[j+M] * sc1 + base[j+M]) * post_mult
            post_row = fx.slice(post_buf, (token, None))
            post_row_div = fx.logical_divide(post_row, fx.make_layout(1, 1))
            for j in range_constexpr(M):
                z = mixes[j + M] * sc1 + base_vals[j + M]
                neg_z = fx.Float32(0.0) - z
                e = fx.Float32(fxmath.exp(neg_z))
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + e)
                ov = sig * fx.Float32(post_mult)
                outr = fx.memref_alloca(scalar_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [ov]), outr)
                fx.copy_atom_call(copy_atom_32, outr, fx.slice(post_row_div, (None, j)))

            # ---- Write comb[j, k] = mixes[j*M + k + 2*M] * sc2 + base[...]
            cb_plane = fx.slice(cb_buf, (token, None, None))
            for j in range_constexpr(M):
                cb_row = fx.slice(cb_plane, (j, None))
                cb_row_div = fx.logical_divide(cb_row, fx.make_layout(1, 1))
                for k in range_constexpr(M):
                    idx = j * M + k + 2 * M
                    z = mixes[idx] * sc2 + base_vals[idx]
                    outr = fx.memref_alloca(scalar_ty, s_lay)
                    fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [z]), outr)
                    fx.copy_atom_call(copy_atom_32, outr, fx.slice(cb_row_div, (None, k)))

    @flyc.jit
    def launch(
        gemm_out_mul:    fx.Tensor,
        gemm_out_sqrsum: fx.Tensor,
        hc_scale:        fx.Tensor,
        hc_base:         fx.Tensor,
        pre_mix:         fx.Tensor,
        post_mix:        fx.Tensor,
        comb_unnormed:   fx.Tensor,
        num_tokens:      fx.Int32,
        stream:          fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        kernel(
            gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
            pre_mix, post_mix, comb_unnormed, num_tokens,
        ).launch(grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    def runner(
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
        pre_mix, post_mix, comb_unnormed, num_tokens,
    ):
        if num_tokens == 0:
            return
        launch(
            gemm_out_mul.detach(), gemm_out_sqrsum.detach(),
            hc_scale.detach(), hc_base.detach(),
            pre_mix.detach(), post_mix.detach(), comb_unnormed.detach(),
            num_tokens,
        )

    return runner


_KERNEL_CACHE: dict[tuple, object] = {}


def get_norm_split_kernel(
    mhc_mult: int,
    hidden_size: int,
    n_splits: int,
    rms_eps: float,
    pre_eps: float,
    post_mult: float,
):
    key = (
        int(mhc_mult), int(hidden_size), int(n_splits),
        float(rms_eps), float(pre_eps), float(post_mult),
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _build_norm_split_kernel(*key)
    return _KERNEL_CACHE[key]
