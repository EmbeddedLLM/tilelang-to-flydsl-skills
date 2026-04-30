"""normalize_weight: row-local L1 normalisation of top-k routing weights.

One thread per token row.  Each thread loads its full K-vector in a single
BufferCopy, sums over K, and writes the normalised vector back.  Matches the
TileKernels semantics including the `1e-20` epsilon bias on the denominator.
"""

import os
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.vector import ReductionOp, full

from fly_tile_kernels._flydsl_helpers import pick_buffer_copy_atom, make_register_memref


def get_normalize_weight_kernel(num_topk: int):
    NUM_THREADS = 128
    copy_op, n_atoms = pick_buffer_copy_atom(fx.Float32, num_topk)
    if n_atoms != 1:
        raise NotImplementedError(
            f"normalize_weight: num_topk={num_topk} requires {n_atoms} BufferCopy atoms; "
            f"only 1-atom widths supported in this port. Implement multi-atom unrolling if needed."
        )

    @flyc.kernel
    def normalize_weight_kernel(
        topk_weights:       fx.Tensor,
        denominator:        fx.Tensor,
        normalized_weights: fx.Tensor,
        num_tokens:         fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        row = bid * NUM_THREADS + tid

        x_buf = fx.rocdl.make_buffer_tensor(topk_weights)
        d_buf = fx.rocdl.make_buffer_tensor(denominator)
        n_buf = fx.rocdl.make_buffer_tensor(normalized_weights)

        reg_ty, reg_lay = make_register_memref(fx.Float32, num_topk)
        scalar_ty, scalar_lay = make_register_memref(fx.Float32, 1)

        copy_atom_vec = fx.make_copy_atom(copy_op, fx.Float32)
        copy_atom_scalar = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        # 1-D denominator: divide into per-element tiles so we can use
        # `fx.slice(d_div, (None, row))` to get a single-element memref view.
        d_div = fx.logical_divide(d_buf, fx.make_layout(1, 1))

        if row < num_tokens:
            # Load: slice 2-D `x_buf` to row, divide by num_topk to get a 1-tile view.
            r_in = fx.memref_alloca(reg_ty, reg_lay)
            row_view = fx.slice(x_buf, (row, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(num_topk, 1))
            fx.copy_atom_call(copy_atom_vec, fx.slice(row_div, (None, 0)), r_in)

            v_in = fx.memref_load_vec(r_in)
            sum_v = v_in.reduce(ReductionOp.ADD)
            sum_with_eps = sum_v + fx.Float32(1e-20)

            # Store denominator[row] = sum_with_eps via a single-element register memref.
            d_reg = fx.memref_alloca(scalar_ty, scalar_lay)
            fx.memref_store_vec(full(1, sum_with_eps, fx.Float32), d_reg)
            fx.copy_atom_call(copy_atom_scalar, d_reg, fx.slice(d_div, (None, row)))

            # Store normalised weights[row, :] = v_in / sum_with_eps.
            v_out = v_in / sum_with_eps
            r_out = fx.memref_alloca(reg_ty, reg_lay)
            fx.memref_store_vec(v_out, r_out)
            n_view = fx.slice(n_buf, (row, None))
            n_div = fx.logical_divide(n_view, fx.make_layout(num_topk, 1))
            fx.copy_atom_call(copy_atom_vec, r_out, fx.slice(n_div, (None, 0)))

    @flyc.jit
    def launch(
        topk_weights:       fx.Tensor,
        denominator:        fx.Tensor,
        normalized_weights: fx.Tensor,
        num_tokens:         fx.Int32,
        stream:             fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        normalize_weight_kernel(
            topk_weights, denominator, normalized_weights, num_tokens,
        ).launch(grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize top-k routing weights so each token's weights sum to 1 (with 1e-20 bias).

    Args:
        topk_weights: FP32 tensor of shape (num_tokens, num_topk).

    Returns:
        (denominator, normalized_weights) — denominator (num_tokens,) and
        normalized_weights (num_tokens, num_topk).
    """
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32
    num_tokens, num_topk = topk_weights.shape

    kernel = get_normalize_weight_kernel(num_topk)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        pass

    denominator = torch.empty((num_tokens,), dtype=torch.float32, device="cuda")
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device="cuda")
    if num_tokens > 0:
        kernel(topk_weights, denominator, normalized_weights, num_tokens)
    return denominator, normalized_weights
