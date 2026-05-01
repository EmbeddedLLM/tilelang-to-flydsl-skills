"""mhc_head_compute_mix: per-element sigmoid + scale + base + eps.

Forward (per (token, j)):
    out[token, j] = sigmoid(input[token, j] * mhc_scale[0] + mhc_base[j])
                  + mhc_pre_eps

Backward (per (token, j)):
    f                    = sigmoid(input * mhc_scale + mhc_base) (recompute)
    d                    = f * (1 - f) * output_grad
    input_grad[t, j]     = d * mhc_scale[0]
    mhc_scale_grad      += sum over (t, j) of d * input[t, j]
    mhc_base_grad[j]    += sum over t of d

The forward kernel is a real FlyDSL elementwise kernel: one thread per
(token, j) element, mhc_mult tiled by the inner dimension of the block,
sigmoid implemented via 1 / (1 + exp(-x)) using ``flydsl.expr.math.exp``.

The backward kernel is kept as a torch fallback because it requires
cross-thread atomic reductions for ``mhc_scale_grad`` (scalar) and
``mhc_base_grad`` (mhc_mult-vector).  The wrapper expects per-SM partials
(num_sms × mhc rows) which the host then sums via ``.sum(0)`` — our
fallback parks the full reduction in slot 0.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr, math as fxmath
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def _mhc_head_compute_mix_fwd(mhc_mult: int, mhc_pre_eps: float, token_block_size: int = 32):
    """Forward kernel: per-element sigmoid + scale + base + eps."""
    mhc = mhc_mult
    eps = mhc_pre_eps

    @flyc.kernel
    def fwd_kernel(
        input_mix:  fx.Tensor,   # (n, mhc) fp32
        mhc_scale:  fx.Tensor,   # (1,)     fp32
        mhc_base:   fx.Tensor,   # (mhc,)   fp32
        output_mix: fx.Tensor,   # (n, mhc) fp32
        num_tokens: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        token = bid * NUM_THREADS + tid

        x_buf = fx.rocdl.make_buffer_tensor(input_mix)
        s_buf = fx.rocdl.make_buffer_tensor(mhc_scale)
        b_buf = fx.rocdl.make_buffer_tensor(mhc_base)
        o_buf = fx.rocdl.make_buffer_tensor(output_mix)

        scalar_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1 = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(token < num_tokens):
            # Load mhc_scale[0] once per thread.
            s_div = fx.logical_divide(s_buf, fx.make_layout(1, 1))
            s_reg = fx.memref_alloca(scalar_ty, s_lay)
            fx.copy_atom_call(copy_atom_32, fx.slice(s_div, (None, 0)), s_reg)
            scale_val = fx.Float32(
                fxvec.extract(fx.memref_load_vec(s_reg), static_position=[0])
            )

            # x_row = input_mix[token, :] view, b = mhc_base view.
            x_row = fx.slice(x_buf, (token, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
            o_row = fx.slice(o_buf, (token, None))
            o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
            b_div = fx.logical_divide(b_buf, fx.make_layout(1, 1))

            for j in range_constexpr(mhc):
                x_reg = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(x_row_div, (None, j)), x_reg)
                x_val = fx.Float32(
                    fxvec.extract(fx.memref_load_vec(x_reg), static_position=[0])
                )

                b_reg = fx.memref_alloca(scalar_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(b_div, (None, j)), b_reg)
                b_val = fx.Float32(
                    fxvec.extract(fx.memref_load_vec(b_reg), static_position=[0])
                )

                # sigmoid(x * scale + base) + eps  via 1 / (1 + exp(-z)).
                z = x_val * scale_val + b_val
                neg_z = fx.Float32(0.0) - z
                e = fx.Float32(fxmath.exp(neg_z))
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + e)
                out_v = sig + fx.Float32(eps)

                o_reg = fx.memref_alloca(scalar_ty, s_lay)
                o_vec = fxvec.from_elements(vec_ty_1, [out_v])
                fx.memref_store_vec(o_vec, o_reg)
                fx.copy_atom_call(copy_atom_32, o_reg, fx.slice(o_row_div, (None, j)))

    @flyc.jit
    def launch(
        input_mix:  fx.Tensor,
        mhc_scale:  fx.Tensor,
        mhc_base:   fx.Tensor,
        output_mix: fx.Tensor,
        num_tokens: fx.Int32,
        stream:     fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        fwd_kernel(input_mix, mhc_scale, mhc_base, output_mix, num_tokens).launch(
            grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
        )

    def runner(input_mix, mhc_scale, mhc_base, output_mix):
        # input_mix: (n, mhc) fp32; mhc_scale: (1,) fp32; mhc_base: (mhc,) fp32
        # output_mix: (n, mhc) fp32 -- written in place
        n = input_mix.shape[0]
        if n == 0:
            return
        launch(input_mix.detach(), mhc_scale.detach(), mhc_base.detach(),
               output_mix.detach(), n)

    return runner


def _mhc_head_compute_mix_bwd(mhc_mult: int, token_block_size: int = 32, num_sms: int = 1):
    """Backward kernel: torch fallback (port pending — needs partial-sum
    reductions across blocks for ``mhc_scale_grad`` and ``mhc_base_grad``).

    Returns a callable matching the wrapper's expected signature.
    """
    def runner(
        output_mix_grad,           # (n, mhc) fp32
        input_mix,                 # (n, mhc) fp32
        mhc_scale,                 # (1,) fp32
        mhc_base,                  # (mhc,) fp32
        input_mix_grad,            # (n, mhc) fp32  out
        mhc_scale_grad_partial,    # (num_sms, 1) fp32  out
        mhc_base_grad_partial,     # (num_sms, mhc) fp32  out
    ):
        f = torch.sigmoid(input_mix * mhc_scale + mhc_base)
        d = f * (1.0 - f) * output_mix_grad

        torch.mul(d, mhc_scale, out=input_mix_grad)

        mhc_scale_grad_partial.zero_()
        mhc_base_grad_partial.zero_()
        mhc_scale_grad_partial[0, 0] = (d * input_mix).sum()
        mhc_base_grad_partial[0, :] = d.sum(dim=0)
    return runner
