"""mhc_post: per-token output is x[t, h] times the post mix plus a (mhc, mhc)
mix-of-residuals contraction.

Forward (per token t, mhc-output index ``mhco``, hidden ``h``):
    out[t, mhco, h] = post_mix[t, mhco] * x[t, h]
                    + sum_{mhci} comb_mix[t, mhci, mhco] * residual[t, mhci, h]

Backward derives full grads w.r.t. (x, residual, post_mix, comb_mix).

The forward kernel is a real FlyDSL kernel: one thread per (token, h)
hidden-element, mhc-output dimensions unrolled inside the thread.

The backward is kept as a torch fallback because computing
``d_post_mix`` and ``d_comb_mix`` requires reductions over the h
dimension; the torch fallback uses ``torch.autograd.grad`` to derive all
four gradients in one pass.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


def _dyn(x):
    return x


def _mhc_post_fwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024):
    """Forward kernel: one thread per (token, h) — mhc output dim unrolled."""
    h = hidden
    M = mhc
    BLK_H = 256

    @flyc.kernel
    def fwd_kernel(
        a:          fx.Tensor,   # (n, mhc, mhc) fp32  — comb_res_mix
        b:          fx.Tensor,   # (n, mhc, h)   bf16  — residual
        c:          fx.Tensor,   # (n, mhc)      fp32  — post_layer_mix
        d:          fx.Tensor,   # (n, h)        bf16  — x
        out:        fx.Tensor,   # (n, mhc, h)   bf16  — output
        num_tokens: fx.Int32,
    ):
        token = fx.block_idx.x
        bid_h = fx.block_idx.y
        tid = fx.thread_idx.x
        h_idx = bid_h * BLK_H + tid

        a_buf = fx.rocdl.make_buffer_tensor(a)
        b_buf = fx.rocdl.make_buffer_tensor(b)
        c_buf = fx.rocdl.make_buffer_tensor(c)
        d_buf = fx.rocdl.make_buffer_tensor(d)
        o_buf = fx.rocdl.make_buffer_tensor(out)

        bf16_reg_ty = fx.MemRefType.get(
            fx.BFloat16.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        fp32_reg_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        vec_ty_1_bf = ir.VectorType.get([1], fx.BFloat16.ir_type)

        if _dyn(h_idx < h):
            # Load c[token, 0..M-1] (post_layer_mix) — M fp32.
            c_row = fx.slice(c_buf, (token, None))
            c_row_div = fx.logical_divide(c_row, fx.make_layout(1, 1))
            c_vals = []
            for k in range_constexpr(M):
                r = fx.memref_alloca(fp32_reg_ty, s_lay)
                fx.copy_atom_call(copy_atom_32, fx.slice(c_row_div, (None, k)), r)
                c_vals.append(
                    fx.Float32(fxvec.extract(fx.memref_load_vec(r), static_position=[0]))
                )

            # Load a[token, 0..M-1, 0..M-1] (comb_res_mix) — M*M fp32.
            a_vals = [[None] * M for _ in range(M)]
            for ii in range_constexpr(M):
                a_row = fx.slice(a_buf, (token, ii, None))
                a_row_div = fx.logical_divide(a_row, fx.make_layout(1, 1))
                for jj in range_constexpr(M):
                    r = fx.memref_alloca(fp32_reg_ty, s_lay)
                    fx.copy_atom_call(copy_atom_32, fx.slice(a_row_div, (None, jj)), r)
                    a_vals[ii][jj] = fx.Float32(
                        fxvec.extract(fx.memref_load_vec(r), static_position=[0])
                    )

            # Load d[token, h_idx] (x).
            d_row = fx.slice(d_buf, (token, None))
            d_row_div = fx.logical_divide(d_row, fx.make_layout(1, 1))
            dr = fx.memref_alloca(bf16_reg_ty, s_lay)
            fx.copy_atom_call(copy_atom_16, fx.slice(d_row_div, (None, h_idx)), dr)
            d_val_bf = fxvec.extract(fx.memref_load_vec(dr), static_position=[0])
            d_val = fx.BFloat16(d_val_bf).to(fx.Float32)

            # Load b[token, m, h_idx] for m in 0..M-1 (residual).
            b_vals = []
            for m in range_constexpr(M):
                b_row = fx.slice(b_buf, (token, m, None))
                b_row_div = fx.logical_divide(b_row, fx.make_layout(1, 1))
                br = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.copy_atom_call(copy_atom_16, fx.slice(b_row_div, (None, h_idx)), br)
                bv_bf = fxvec.extract(fx.memref_load_vec(br), static_position=[0])
                b_vals.append(fx.BFloat16(bv_bf).to(fx.Float32))

            # Compute and store mhc outputs.
            for mhco in range_constexpr(M):
                acc = c_vals[mhco] * d_val
                for mhci in range_constexpr(M):
                    acc = acc + a_vals[mhci][mhco] * b_vals[mhci]
                acc_bf = acc.to(fx.BFloat16)

                o_row = fx.slice(o_buf, (token, mhco, None))
                o_row_div = fx.logical_divide(o_row, fx.make_layout(1, 1))
                outr = fx.memref_alloca(bf16_reg_ty, s_lay)
                fx.memref_store_vec(fxvec.from_elements(vec_ty_1_bf, [acc_bf]), outr)
                fx.copy_atom_call(copy_atom_16, outr, fx.slice(o_row_div, (None, h_idx)))

    @flyc.jit
    def launch(
        a:          fx.Tensor,
        b:          fx.Tensor,
        c:          fx.Tensor,
        d:          fx.Tensor,
        out:        fx.Tensor,
        num_tokens: fx.Int32,
        stream:     fx.Stream = fx.Stream(None),
    ):
        gx = num_tokens
        gy = (h + BLK_H - 1) // BLK_H
        fwd_kernel(a, b, c, d, out, num_tokens).launch(
            grid=(gx, gy, 1), block=(BLK_H, 1, 1), stream=stream,
        )

    def runner(a, b, c, d, out):
        n = a.shape[0]
        if n == 0:
            return
        launch(a.detach(), b.detach(), c.detach(), d.detach(), out.detach(), n)

    return runner


def _mhc_post_compute(x, residual, post_layer_mix, comb_res_mix):
    """Pure-torch reference of the forward (used by mhc_post_bwd's autograd
    replay; ``mhc_post_fwd`` calls into the FlyDSL kernel above)."""
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out):
    """Wrapper called by the modeling-layer autograd Function."""
    num_seqs, num_tokens, mhc, hidden = residual.shape

    if out is None:
        out = torch.empty_like(residual)

    runner = _mhc_post_fwd(mhc, hidden)
    n = num_seqs * num_tokens

    runner(
        comb_res_mix.view(n, mhc, mhc),
        residual.view(n, mhc, hidden),
        post_layer_mix.view(n, mhc, 1).squeeze(-1),
        x.view(n, hidden),
        out.view(n, mhc, hidden),
    )
    return out


def mhc_post_bwd(x, residual, post_layer_mix, comb_res_mix, d_o):
    """Returns (d_x, d_residual, d_post_layer_mix, d_comb_res_mix).

    Currently a torch autograd-replay fallback because the full BWD kernel
    needs cross-h reductions for ``d_post_mix`` and ``d_comb_mix``.
    """
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        r_ = residual.detach().requires_grad_(True)
        p_ = post_layer_mix.detach().requires_grad_(True)
        c_ = comb_res_mix.detach().requires_grad_(True)
        out = _mhc_post_compute(x_, r_, p_, c_)
        grads = torch.autograd.grad(out, [x_, r_, p_, c_], d_o)
    return grads
