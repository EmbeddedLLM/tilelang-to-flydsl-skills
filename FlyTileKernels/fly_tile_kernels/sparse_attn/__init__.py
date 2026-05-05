"""sparse_attn: FlyDSL port of tilelang-ascend's DeepseekV4 sparse-flash-attention.

Public entry: ``sparse_attn(q, kv, attn_sink, topk_idxs, scale)``.
"""

from .sparse_attn_kernel import sparse_attn

__all__ = ["sparse_attn"]
