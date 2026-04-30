"""engram_hash: per-(layer, token) n-gram hash → embedding-index lookup.

Each thread handles one (layer, token) pair:

  hash = 0
  for ngram_idx in 0..max_ngram_size:
      hash ^= int64(token_ids[ngram_idx]) * multipliers[layer, ngram_idx]
      if ngram_idx > 0:
          for j in 0..num_embed_table_per_ngram:
              col = (ngram_idx - 1) * num_embed_table_per_ngram + j
              out[layer, token, col] = (hash % vocab_sizes[layer, ngram_idx-1, j]) + offsets[layer, col]

All math runs in int64; the final output is cast to int32.

Layout choice:
- Grid: ``(num_ngram_layers, ceildiv(num_tokens, BLK_M))``.
- Block: ``BLK_M`` threads, one token per thread.

API notes (this FlyDSL build, gfx950): see
``moe/normalize_weight_kernel.py`` for the underlying patterns
(``_dyn(...)`` for dynamic-if; per-element BufferCopy load/store).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector as fxvec, range_constexpr
from flydsl._mlir import ir


BLK_M = 32


def _dyn(x):
    return x


def get_engram_hash_kernel(
    max_ngram_size: int,
    num_ngram_layers: int,
    num_embed_table_per_ngram: int,
):
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram
    K = max_ngram_size

    @flyc.kernel
    def hash_kernel(
        ngram_token_ids: fx.Tensor,   # (num_tokens, K) int32
        multipliers:     fx.Tensor,   # (num_ngram_layers, K) int64
        vocab_sizes:     fx.Tensor,   # (num_ngram_layers, K-1, num_embed_table_per_ngram) int32
        offsets:         fx.Tensor,   # (num_ngram_layers, num_out_cols) int32
        output:          fx.Tensor,   # (num_ngram_layers, num_tokens, num_out_cols) int32
        num_tokens:      fx.Int32,
    ):
        layer = fx.block_idx.x
        bid_y = fx.block_idx.y
        tid = fx.thread_idx.x
        token = bid_y * BLK_M + tid

        x_buf = fx.rocdl.make_buffer_tensor(ngram_token_ids)
        mul_buf = fx.rocdl.make_buffer_tensor(multipliers)
        vs_buf = fx.rocdl.make_buffer_tensor(vocab_sizes)
        off_buf = fx.rocdl.make_buffer_tensor(offsets)
        out_buf = fx.rocdl.make_buffer_tensor(output)

        i32_reg_ty = fx.MemRefType.get(
            fx.Int32.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        i64_reg_ty = fx.MemRefType.get(
            fx.Int64.ir_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register,
        )
        s_lay = fx.make_layout(1, 1)
        copy_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int64)
        vec_ty_1_i32 = ir.VectorType.get([1], fx.Int32.ir_type)

        if _dyn(token < num_tokens):
            # Load token_ids[token, :] (K int32 values).
            x_row = fx.slice(x_buf, (token, None))
            x_row_div = fx.logical_divide(x_row, fx.make_layout(1, 1))
            tok_vals = []
            for k in range_constexpr(K):
                r = fx.memref_alloca(i32_reg_ty, s_lay)
                fx.copy_atom_call(
                    copy_atom_32, fx.slice(x_row_div, (None, k)), r,
                )
                tok_vals.append(
                    fxvec.extract(fx.memref_load_vec(r), static_position=[0]),
                )

            # Load multipliers[layer, :] (K int64 values).
            mul_row = fx.slice(mul_buf, (layer, None))
            mul_row_div = fx.logical_divide(mul_row, fx.make_layout(1, 1))
            mul_vals = []
            for k in range_constexpr(K):
                r = fx.memref_alloca(i64_reg_ty, s_lay)
                fx.copy_atom_call(
                    copy_atom_64, fx.slice(mul_row_div, (None, k)), r,
                )
                mul_vals.append(
                    fxvec.extract(fx.memref_load_vec(r), static_position=[0]),
                )

            # Load offsets[layer, :] (num_out_cols int32 values).
            off_row = fx.slice(off_buf, (layer, None))
            off_row_div = fx.logical_divide(off_row, fx.make_layout(1, 1))
            off_vals = []
            for c in range_constexpr(num_out_cols):
                r = fx.memref_alloca(i32_reg_ty, s_lay)
                fx.copy_atom_call(
                    copy_atom_32, fx.slice(off_row_div, (None, c)), r,
                )
                off_vals.append(
                    fxvec.extract(fx.memref_load_vec(r), static_position=[0]),
                )

            # Load vocab_sizes[layer, :, :] flattened — slice to 1-D first.
            # The 3-D buffer slice (layer, None, None) yields 2-D (K-1, T).
            # Then slice once more (k_idx, None) → 1-D row of length T.
            vs_layer = fx.slice(vs_buf, (layer, None, None))
            vs_vals = [[None] * num_embed_table_per_ngram for _ in range(K - 1)]
            for ki in range_constexpr(K - 1):
                vs_row = fx.slice(vs_layer, (ki, None))
                vs_row_div = fx.logical_divide(vs_row, fx.make_layout(1, 1))
                for j in range_constexpr(num_embed_table_per_ngram):
                    r = fx.memref_alloca(i32_reg_ty, s_lay)
                    fx.copy_atom_call(
                        copy_atom_32, fx.slice(vs_row_div, (None, j)), r,
                    )
                    vs_vals[ki][j] = fxvec.extract(
                        fx.memref_load_vec(r), static_position=[0],
                    )

            # Hash computation.
            hash_v = fx.Int64(0)
            output_vals = [None] * num_out_cols
            for ngram_idx in range_constexpr(K):
                tok_i64 = fx.Int64(tok_vals[ngram_idx])
                mul_i64 = fx.Int64(mul_vals[ngram_idx])
                hash_v = hash_v ^ (tok_i64 * mul_i64)
                if ngram_idx > 0:
                    for j in range_constexpr(num_embed_table_per_ngram):
                        col = (ngram_idx - 1) * num_embed_table_per_ngram + j
                        vs_i64 = fx.Int64(vs_vals[ngram_idx - 1][j])
                        modv = hash_v % vs_i64
                        modv_i32 = fx.Int32(modv)
                        output_vals[col] = modv_i32 + fx.Int32(off_vals[col])

            # Write output[layer, token, :] one int32 at a time.
            out_layer = fx.slice(out_buf, (layer, None, None))  # 2-D (n_tokens, num_out_cols)
            out_row = fx.slice(out_layer, (token, None))         # 1-D (num_out_cols,)
            out_row_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
            for c in range_constexpr(num_out_cols):
                w_reg = fx.memref_alloca(i32_reg_ty, s_lay)
                w_vec = fxvec.from_elements(vec_ty_1_i32, [output_vals[c]])
                fx.memref_store_vec(w_vec, w_reg)
                fx.copy_atom_call(copy_atom_32, w_reg, fx.slice(out_row_div, (None, c)))

    @flyc.jit
    def launch(
        ngram_token_ids: fx.Tensor,
        multipliers:     fx.Tensor,
        vocab_sizes:     fx.Tensor,
        offsets:         fx.Tensor,
        output:          fx.Tensor,
        num_tokens:      fx.Int32,
        stream:          fx.Stream = fx.Stream(None),
    ):
        gx = num_ngram_layers
        gy = (num_tokens + BLK_M - 1) // BLK_M
        hash_kernel(
            ngram_token_ids, multipliers, vocab_sizes, offsets, output, num_tokens,
        ).launch(grid=(gx, gy, 1), block=(BLK_M, 1, 1), stream=stream)

    return launch


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute n-gram hash embedding indices.

    Args:
        ngram_token_ids: (num_tokens, max_ngram_size), int32.
        multipliers: (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_tokens, max_ngram_size = ngram_token_ids.shape
    num_ngram_layers, _, num_embed_table_per_ngram = vocab_sizes.shape
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    output = torch.empty(
        (num_ngram_layers, num_tokens, num_out_cols),
        dtype=torch.int32,
        device=ngram_token_ids.device,
    )
    if num_tokens == 0:
        return output

    kernel = get_engram_hash_kernel(
        max_ngram_size, num_ngram_layers, num_embed_table_per_ngram,
    )
    kernel(ngram_token_ids, multipliers, vocab_sizes, offsets, output, num_tokens)
    return output
