# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for AttentionWithRope in max.nn.attention."""

from __future__ import annotations

from typing import cast
from unittest import mock

import pytest
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Graph, ops
from max.nn.attention import AttentionWithRope, DistributedAttentionWithRope
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
)
from max.nn.rotary_embedding import RotaryEmbedding


def test_attention_with_rope_stacked_qkv_bias_validation() -> None:
    """Tests that AttentionWithRope raises ValueError for stacked_qkv with bias."""
    rope = RotaryEmbedding(
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
    rope = RotaryEmbedding(
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


def test_distributed_attention_with_rope_device_validation() -> None:
    """Tests that DistributedAttentionWithRope raises ValueError for < 2 devices."""
    rope = RotaryEmbedding(
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

    # Test that devices=None raises ValueError.
    with pytest.raises(ValueError, match="Must provide at least 2 devices"):
        DistributedAttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            devices=None,
        )

    # Test that devices=[] raises ValueError.
    with pytest.raises(ValueError, match="Must provide at least 2 devices"):
        DistributedAttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            devices=[],
        )

    # Test that devices=[CPU] raises ValueError.
    with pytest.raises(ValueError, match="Must provide at least 2 devices"):
        DistributedAttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            devices=[DeviceRef.CPU()],
        )

    # Test that CPU devices raises ValueError.
    with pytest.raises(
        ValueError, match="DistributedAttentionWithRope does not support CPU"
    ):
        DistributedAttentionWithRope(
            rope=rope,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=2048,
            kv_params=kv_params,
            devices=[DeviceRef.CPU(), DeviceRef.CPU()],
        )


@mock.patch("max.nn.attention.attention_with_rope.Allreduce")
def test_distributed_attention_with_rope_call_validation(
    allreduce_mock: mock.Mock,
) -> None:
    """Tests input validation in DistributedAttentionWithRope.__call__."""
    rope = RotaryEmbedding(
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

    devices = [DeviceRef("gpu", i) for i in range(2)]
    dist_attn = DistributedAttentionWithRope(
        rope=rope,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        kv_params=kv_params,
        devices=devices,
    )

    # Dummy inputs for __call__
    with Graph(name="test_graph") as g:
        layer_idx = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())
        dummy_tensor = ops.constant(
            0.0, dtype=DType.float32, device=DeviceRef.CPU()
        )
        x = [dummy_tensor, dummy_tensor]
        signal_buffers = [mock.Mock(spec=BufferValue) for _ in devices]
        kv_collections = [
            mock.Mock(spec=PagedKVCacheCollection) for _ in devices
        ]

        # Test wrong number of input_row_offsets
        with pytest.raises(
            ValueError, match="Expected 2 input_row_offsets, got 1"
        ):
            dist_attn(
                layer_idx,
                x,
                cast(list[BufferValue], signal_buffers),
                cast(
                    list[PagedKVCacheCollection],
                    kv_collections,
                ),
                input_row_offsets=[dummy_tensor],
            )

        # Test wrong type in input_row_offsets
        with pytest.raises(
            TypeError,
            match="All elements in input_row_offsets must be TensorValue instances",
        ):
            dist_attn(
                layer_idx,
                x,
                cast(list[BufferValue], signal_buffers),
                cast(
                    list[PagedKVCacheCollection],
                    kv_collections,
                ),
                input_row_offsets=[dummy_tensor, "not-a-tensor-value"],  # type: ignore
            )
