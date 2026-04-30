"""mask_indices_by_tp: per-element int64 expert-index remapping.

For each (token, k) entry of the int64 ``indices`` tensor:

  * If ``value < 0`` → write ``-1``.
  * If ``(value / per_gpu) % num_tp_ranks != tp_rank`` (this token's expert
    belongs to a different TP rank) → write ``-1``.
  * Otherwise: subtract ``tp_rank * per_gpu`` to land in the local
    expert range, then collapse ``per_dp - per_gpu`` per dp_rank, landing
    in the local-only range; if the result is negative (only happens on
    certain dp_rank/per_dp combos), write ``-1``.

API notes (this FlyDSL build, gfx950):
- Branching condition uses ``Boolean.select`` chained against -1.  This
  avoids the AST rewriter's `or`/`and` no-rewrite restrictions.
- int64 elements are loaded/stored via ``BufferCopy64b`` against the
  Int64 element type — no integer-LDS-atomic primitive needed for this
  kernel.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec
from flydsl._mlir import ir


NUM_THREADS = 128


def _dyn(x):
    return x


def get_mask_indices_by_tp_kernel(num_topk: int):
    @flyc.kernel
    def mask_kernel(
        indices:        fx.Tensor,   # (num_tokens, num_topk) int64
        masked:         fx.Tensor,   # same shape int64
        per_gpu:        fx.Int32,
        per_dp:         fx.Int32,
        num_tp_ranks:   fx.Int32,
        tp_rank:        fx.Int32,
        num_elems:      fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        idx = bid * NUM_THREADS + tid

        in_buf = fx.rocdl.make_buffer_tensor(indices)
        out_buf = fx.rocdl.make_buffer_tensor(masked)

        scalar_ty = fx.MemRefType.get(
            fx.Int64.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int64)

        vec_ty_1_i64 = ir.VectorType.get([1], fx.Int64.ir_type)

        if _dyn(idx < num_elems):
            # Decompose flat idx into (token, k).
            token = idx // num_topk
            k = idx % num_topk

            # Load via 2-step slice: row → tile-by-1 → element.
            in_row = fx.slice(in_buf, (token, None))
            in_row_div = fx.logical_divide(in_row, fx.make_layout(1, 1))

            in_reg = fx.memref_alloca(scalar_ty, s_lay)
            src = fx.slice(in_row_div, (None, k))
            fx.copy_atom_call(copy_atom_64, src, in_reg)
            value = fxvec.extract(fx.memref_load_vec(in_reg), static_position=[0])
            value = fx.Int64(value)

            # Promote runtime i32 args to i64 for the math.
            per_gpu_i64 = fx.Int64(per_gpu)
            per_dp_i64 = fx.Int64(per_dp)
            num_tp_ranks_i64 = fx.Int64(num_tp_ranks)
            tp_rank_i64 = fx.Int64(tp_rank)

            # Compute the mod check (always — if value<0 the result is a
            # junk number we mask away with a select below).
            v_mod = (value // per_gpu_i64) % num_tp_ranks_i64

            # Compute the unmasked output value.
            v2 = value - tp_rank_i64 * per_gpu_i64
            dp_rank = v2 // per_dp_i64
            v3 = v2 - dp_rank * (per_dp_i64 - per_gpu_i64)

            # Chain selects: if any of {value<0, mod-mismatch, v3<0} → -1,
            # else v3.
            neg1 = fx.Int64(-1)
            cond_neg = value < fx.Int64(0)
            cond_mod = v_mod != tp_rank_i64
            cond_v3 = v3 < fx.Int64(0)
            out_v = cond_neg.select(
                neg1,
                cond_mod.select(
                    neg1,
                    cond_v3.select(neg1, v3),
                ),
            )

            out_reg = fx.memref_alloca(scalar_ty, s_lay)
            out_vec = fxvec.from_elements(vec_ty_1_i64, [out_v])
            fx.memref_store_vec(out_vec, out_reg)
            out_row = fx.slice(out_buf, (token, None))
            out_row_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
            dst = fx.slice(out_row_div, (None, k))
            fx.copy_atom_call(copy_atom_64, out_reg, dst)

    @flyc.jit
    def launch(
        indices:        fx.Tensor,
        masked:         fx.Tensor,
        per_gpu:        fx.Int32,
        per_dp:         fx.Int32,
        num_tp_ranks:   fx.Int32,
        tp_rank:        fx.Int32,
        num_elems:      fx.Int32,
        stream:         fx.Stream = fx.Stream(None),
    ):
        gx = (num_elems + NUM_THREADS - 1) // NUM_THREADS
        mask_kernel(
            indices, masked, per_gpu, per_dp, num_tp_ranks, tp_rank, num_elems,
        ).launch(grid=(gx, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


def mask_indices_by_tp(
    indices: torch.Tensor,
    n: int,
    num_ep_ranks: int,
    tp_rank: int,
    num_tp_ranks: int,
) -> torch.Tensor:
    """Mask expert indices to keep only those belonging to the given TP rank.

    Args:
        indices: Expert index tensor of shape (num_tokens, num_topk), int64.
        n: Total number of experts across all ranks.
        num_ep_ranks: Expert-parallelism size.
        tp_rank: Tensor-parallelism rank of the current device.
        num_tp_ranks: Tensor-parallelism size.

    Returns:
        Masked index tensor with non-local experts set to -1 and local
        indices remapped to the local expert range.
    """
    assert indices.is_contiguous()
    num_tokens, num_topk = indices.shape
    per_gpu = n // num_ep_ranks
    per_dp = num_tp_ranks * per_gpu

    masked = torch.empty_like(indices)
    if num_tokens == 0:
        return masked

    kernel = get_mask_indices_by_tp_kernel(num_topk)
    num_elems = num_tokens * num_topk
    kernel(indices, masked, per_gpu, per_dp, num_tp_ranks, tp_rank, num_elems)
    return masked
