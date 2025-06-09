# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for AttentionWithRope in max.nn.attention."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.attention import AttentionWithRope
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy
from max.nn.rotary_embedding import OptimizedRotaryEmbedding


def test_attention_with_rope_stacked_qkv_bias_validation() -> None:
    """Tests that AttentionWithRope raises ValueError for stacked_qkv with bias."""
    rope = OptimizedRotaryEmbedding(
        dim=64,
        n_heads=32,
        theta=10000.0,
        max_seq_len=2048,
        device=DeviceRef.CPU(),
    )

    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
        dtype=DType.float32,
    )

    # Test that stacked_qkv=True with has_bias=True raises ValueError.
    with pytest.raises(
        ValueError, match="Bias is not supported with stacked qkv"
    ):
        AttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            stacked_qkv=True,
            has_bias=True,
        )


def test_attention_with_rope_clip_qkv_validation() -> None:
    """Tests that AttentionWithRope raises ValueError for stacked_qkv with clip_qkv."""
    rope = OptimizedRotaryEmbedding(
        dim=64,
        n_heads=32,
        theta=10000.0,
        max_seq_len=2048,
        device=DeviceRef.CPU(),
    )

    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
        dtype=DType.float32,
    )

    # Test that stacked_qkv=True with clip_qkv raises ValueError.
    with pytest.raises(
        ValueError, match="`clip_qkv` not yet supported when `stack_qkv=True`"
    ):
        AttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            stacked_qkv=True,
            clip_qkv=1.0,
        )
