"""normalize_weight: row-local L1 normalisation of top-k routing weights.

One thread per token row.  Each thread loads its full K-vector via per-element
BufferCopy32b atoms (works for any num_topk), folds them into a sum starting
at 1e-20 to match the torch reference's left-fold order bit-exactly, and
writes back the per-element normalised values.

API notes (this FlyDSL build, gfx950):
- AST rewriter only treats an `if` as dynamic if the test contains a Call.
  Wrap simple comparisons in `_dyn(...)` so they are treated dynamically.
- Vector ops want raw `ir.Value` operands; Numeric wrappers need
  `.ir_value()` when handed to `vector.broadcast` / `reduction(acc=...)`.
- A 1-element register memref `(1, 1)` loads as a 1-D `vector<1xf32>`;
  `vector.extract(..., static_position=[0])` retrieves the scalar.
"""

import os
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


def _dyn(x):
    """Identity passthrough used to make the AST rewriter treat a comparison
    as a dynamic condition (it only triggers on `if`-tests that contain a
    Call node in their AST)."""
    return x


def get_normalize_weight_kernel(num_topk: int):
    NUM_THREADS = 128

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

        scalar_ty = fx.MemRefType.get(
            fx.Float32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        d_div = fx.logical_divide(d_buf, fx.make_layout(1, 1))
        vec_ty_1 = ir.VectorType.get([1], fx.Float32.ir_type)

        if _dyn(row < num_tokens):
            row_view = fx.slice(x_buf, (row, None))
            row_div_1 = fx.logical_divide(row_view, fx.make_layout(1, 1))
            n_view = fx.slice(n_buf, (row, None))
            n_div_1 = fx.logical_divide(n_view, fx.make_layout(1, 1))

            # Load each topk weight one element at a time, folding into `acc`
            # in the same order the torch reference uses (1e-20 + w[0] + ...).
            elem_regs = []
            acc = fx.Float32(1e-20)
            for k in range_constexpr(num_topk):
                elem_reg = fx.memref_alloca(scalar_ty, s_lay)
                src = fx.slice(row_div_1, (None, k))
                fx.copy_atom_call(copy_atom_32, src, elem_reg)
                elem_val = fxvec.extract(
                    fx.memref_load_vec(elem_reg), static_position=[0],
                )
                acc = acc + fx.Float32(elem_val)
                elem_regs.append(elem_reg)
            sum_with_eps = acc

            # Write denominator[row].
            d_reg = fx.memref_alloca(scalar_ty, s_lay)
            d_vec = fxvec.from_elements(vec_ty_1, [sum_with_eps])
            fx.memref_store_vec(d_vec, d_reg)
            fx.copy_atom_call(copy_atom_32, d_reg, fx.slice(d_div, (None, row)))

            # Write normalized_weights[row, k] = w[k] / sum_with_eps for each k.
            for k in range_constexpr(num_topk):
                e_val = fxvec.extract(
                    fx.memref_load_vec(elem_regs[k]), static_position=[0],
                )
                e_norm = fx.Float32(e_val) / sum_with_eps
                e_reg = fx.memref_alloca(scalar_ty, s_lay)
                e_vec = fxvec.from_elements(vec_ty_1, [e_norm])
                fx.memref_store_vec(e_vec, e_reg)
                dst = fx.slice(n_div_1, (None, k))
                fx.copy_atom_call(copy_atom_32, e_reg, dst)

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
