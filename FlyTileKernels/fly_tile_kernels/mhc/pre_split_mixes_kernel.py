"""mhc_pre_split_mixes: split a single (n, 2*mhc + mhc^2) input into three
output tensors with elementwise sigmoid-or-linear transforms.

Forward, per token ``t`` and per index ``j``:
    pre[t, j]   = sigmoid(input[t, j]            * scale[0] + base[j])           + eps     for j in [0, mhc)
    post[t, j]  = sigmoid(input[t, j+mhc]        * scale[1] + base[j+mhc])       * post_mult  for j in [0, mhc)
    comb[t, j]  = input[t, j+2*mhc]              * scale[2] + base[j+2*mhc]                   for j in [0, mhc^2)

The forward kernel is a real FlyDSL elementwise kernel: one thread per
token row, mhc + mhc + mhc^2 outputs computed via constexpr loops.

The backward is a torch fallback because mhc_scale_grad / mhc_base_grad
need cross-block reductions (per-SM partial buffers); we compute the
exact reduction in torch and park it in slot 0 of the partials.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, math as fxmath
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def _mhc_pre_split_mixes_fwd(
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int = 32,
    dtype=torch.float32,
):
    """Forward kernel — one thread per token, all elements of all 3 outputs."""
    mhc = mhc_mult
    mhc2 = mhc * mhc
    eps = mhc_pre_eps
    post_mult = mhc_post_mult_value

    @flyc.kernel
    def fwd_kernel(
        input_mixes:    fx.Tensor,   # (n, 2*mhc + mhc^2) fp32
        mhc_scale:      fx.Tensor,   # (3,)               fp32
        mhc_base:       fx.Tensor,   # (2*mhc + mhc^2,)   fp32
        pre_layer_mix:  fx.Tensor,   # (n, mhc)           fp32
        post_layer_mix: fx.Tensor,   # (n, mhc)           fp32
        comb_res_mix:   fx.Tensor,   # (n, mhc^2)         fp32
        num_tokens:     fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        x_buf = fx.rocdl.make_buffer_tensor(input_mixes)
        s_buf = fx.rocdl.make_buffer_tensor(mhc_scale)
        b_buf = fx.rocdl.make_buffer_tensor(mhc_base)
        pre_buf = fx.rocdl.make_buffer_tensor(pre_layer_mix)
        post_buf = fx.rocdl.make_buffer_tensor(post_layer_mix)
        comb_buf = fx.rocdl.make_buffer_tensor(comb_res_mix)

        scalar_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1 = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            # Load mhc_scale[0..2].
            s_div = fx.logical_divide(s_buf, fx.make_layout(1, 1))
            b_div = fx.logical_divide(b_buf, fx.make_layout(1, 1))
            scale_vals = []
            for k in range_constexpr(3):
                r = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(s_div, (None, k)), r)
                scale_vals.append(
                    fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                )
            sc0, sc1, sc2 = scale_vals

            # Slice tensor row views.
            x_row = fx.slice(x_buf, (token, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
            pre_row = fx.slice(pre_buf, (token, None))
            pre_row_div = fx.logical_divide(pre_row, fx.make_layout(1, 1))
            post_row = fx.slice(post_buf, (token, None))
            post_row_div = fx.logical_divide(post_row, fx.make_layout(1, 1))
            comb_row = fx.slice(comb_buf, (token, None))
            comb_row_div = fx.logical_divide(comb_row, fx.make_layout(1, 1))

            # ----- pre: sigmoid(x[j] * sc0 + base[j]) + eps -----
            for j in range_constexpr(mhc):
                xr = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(x_row_div, (None, j)), xr)
                xv = fx.Float32(fxvec.extract(fx.memref_load_vec(xr), static_position=[0]))
                br = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(b_div, (None, j)), br)
                bv = fx.Float32(fxvec.extract(fx.memref_load_vec(br), static_position=[0]))
                z = xv * sc0 + bv
                neg_z = fx.Float32(0.0) - z
                e = fx.Float32(fxmath.exp(neg_z))
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + e)
                ov = sig + fx.Float32(eps)
                outr = fx.memref_alloca(scalar_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [ov]), outr)
                fx.copy_atom_call(copy_atom_32, outr, fx.slice(pre_row_div, (None, j)))

            # ----- post: sigmoid(x[j+mhc] * sc1 + base[j+mhc]) * post_mult -----
            for j in range_constexpr(mhc):
                jx = j + mhc
                xr = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(x_row_div, (None, jx)), xr)
                xv = fx.Float32(fxvec.extract(fx.memref_load_vec(xr), static_position=[0]))
                br = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(b_div, (None, jx)), br)
                bv = fx.Float32(fxvec.extract(fx.memref_load_vec(br), static_position=[0]))
                z = xv * sc1 + bv
                neg_z = fx.Float32(0.0) - z
                e = fx.Float32(fxmath.exp(neg_z))
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + e)
                ov = sig * fx.Float32(post_mult)
                outr = fx.memref_alloca(scalar_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [ov]), outr)
                fx.copy_atom_call(copy_atom_32, outr, fx.slice(post_row_div, (None, j)))

            # ----- comb: x[j+2*mhc] * sc2 + base[j+2*mhc] -----
            for j in range_constexpr(mhc2):
                jx = j + 2 * mhc
                xr = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(x_row_div, (None, jx)), xr)
                xv = fx.Float32(fxvec.extract(fx.memref_load_vec(xr), static_position=[0]))
                br = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(b_div, (None, jx)), br)
                bv = fx.Float32(fxvec.extract(fx.memref_load_vec(br), static_position=[0]))
                ov = xv * sc2 + bv
                outr = fx.memref_alloca(scalar_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1, [ov]), outr)
                fx.copy_atom_call(copy_atom_32, outr, fx.slice(comb_row_div, (None, j)))

    @flyc.jit
    def launch(
        input_mixes:    fx.Tensor,
        mhc_scale:      fx.Tensor,
        mhc_base:       fx.Tensor,
        pre_layer_mix:  fx.Tensor,
        post_layer_mix: fx.Tensor,
        comb_res_mix:   fx.Tensor,
        num_tokens:     fx.Int32,
        stream:         fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        fwd_kernel(
            input_mixes, mhc_scale, mhc_base,
            pre_layer_mix, post_layer_mix, comb_res_mix,
            num_tokens,
        ).launch(grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    def runner(input_mixes, mhc_scale, mhc_base, pre_layer_mix, post_layer_mix, comb_res_mix):
        n = input_mixes.shape[0]
        if n == 0:
            return
        launch(
            input_mixes.detach(), mhc_scale.detach(), mhc_base.detach(),
            pre_layer_mix.detach(), post_layer_mix.detach(), comb_res_mix.detach(),
            n,
        )

    return runner


def _mhc_pre_split_mixes_bwd(
    mhc_mult: int,
    mhc_post_mult_value: float,
    token_block_size: int = 32,
    num_sms: int = 1,
    dtype=torch.float32,
):
    """Backward kernel: torch fallback (port pending — needs partial-sum
    reductions across blocks for ``mhc_scale_grad`` and ``mhc_base_grad``).

    The wrapper expects per-SM partial buffers and sums them with .sum(0).
    We compute the exact reduction in torch and park it in slot 0.
    """
    def runner(
        pre_layer_mix_grad,        # (N, mhc)
        post_layer_mix_grad,       # (N, mhc)
        comb_res_mix_grad,         # (N, mhc^2)
        input_mixes,               # (N, 2*mhc + mhc^2)
        post_layer_mix,            # (N, mhc), saved fwd output (sigmoid * post_mult)
        mhc_scale,                 # (3,)
        mhc_base,                  # (2*mhc + mhc^2,)
        input_mixes_grad,          # (N, 2*mhc + mhc^2) out
        mhc_scale_grad_partial,    # (num_sms, 3) out
        mhc_base_grad_partial,     # (num_sms, 2*mhc + mhc^2) out
    ):
        m = mhc_mult
        f_pre = torch.sigmoid(input_mixes[:, :m] * mhc_scale[0] + mhc_base[:m])
        d_pre = pre_layer_mix_grad * f_pre * (1.0 - f_pre)
        d_post = post_layer_mix_grad * post_layer_mix * (1.0 - post_layer_mix / mhc_post_mult_value)
        d_comb = comb_res_mix_grad

        input_mixes_grad[:, :m] = d_pre * mhc_scale[0]
        input_mixes_grad[:, m:2 * m] = d_post * mhc_scale[1]
        input_mixes_grad[:, 2 * m:] = d_comb * mhc_scale[2]

        mhc_scale_grad_partial.zero_()
        mhc_base_grad_partial.zero_()
        mhc_scale_grad_partial[0, 0] = (d_pre * input_mixes[:, :m]).sum()
        mhc_scale_grad_partial[0, 1] = (d_post * input_mixes[:, m:2 * m]).sum()
        mhc_scale_grad_partial[0, 2] = (d_comb * input_mixes[:, 2 * m:]).sum()
        mhc_base_grad_partial[0, :m] = d_pre.sum(dim=0)
        mhc_base_grad_partial[0, m:2 * m] = d_post.sum(dim=0)
        mhc_base_grad_partial[0, 2 * m:] = d_comb.sum(dim=0)
    return runner
