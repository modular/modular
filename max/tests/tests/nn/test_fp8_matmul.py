# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for FP8 matmul kernels in max.nn.kernels."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import (
    dynamic_scaled_matmul,
    fused_qkv_ragged_matmul_scaled_float8,
)
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
)
from max.nn.kv_cache.paged_cache.paged_cache import PagedKVCacheCollectionType


class DynamicScaledMatmul:
    return_type: DType
    """Return type of the `dynamic_scaled_matmul` custom op."""

    def __init__(self, return_type: DType) -> None:
        self.return_type = return_type

    def __call__(
        self,
        a: TensorValue,
        b: TensorValue,
        a_scales: TensorValue,
        b_scales: TensorValue,
    ) -> TensorValue:
        return dynamic_scaled_matmul(
            a, b, a_scales, b_scales, out_type=self.return_type
        )


def test_dynamic_scaled_matmul_rowwise():
    """Tests dynamic_scaled_matmul with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "dynamic_scaled_matmul",
        input_types=[
            # a
            TensorType(DType.float8_e4m3fn, shape=(2, 4), device=device),
            # b
            TensorType(DType.float8_e4m3fn, shape=(3, 4), device=device),
            # a_scales
            TensorType(DType.bfloat16, shape=(2, 1), device=device),
            # b_scales
            TensorType(DType.bfloat16, shape=(3, 1), device=device),
        ],
    ) as graph:
        a, b, a_scales, b_scales = (inp.tensor for inp in graph.inputs)

        # Test with row-wise weight scales.
        output_rowwise = dynamic_scaled_matmul(
            a, b, a_scales, b_scales, out_type=DType.bfloat16
        )
        assert output_rowwise.shape == [2, 3]
        assert output_rowwise.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "a_dtype, b_dtype, a_scales_dtype, b_scales_dtype, err_msg_part",
    [
        # a.dtype != b.dtype
        (
            DType.float16,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a.dtype != b.dtype
        (
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            "scales dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float16,
            "scales dtypes",
        ),
    ],
)
def test_dynamic_scaled_matmul_dtype_mismatch(
    a_dtype: DType,
    b_dtype: DType,
    a_scales_dtype: DType,
    b_scales_dtype: DType,
    err_msg_part: str,
) -> None:
    """Tests dtype mismatches."""
    device = DeviceRef.CPU()
    with pytest.raises(TypeError, match=err_msg_part):
        Graph(
            "dynamic_scaled_matmul",
            forward=DynamicScaledMatmul(return_type=DType.bfloat16),
            input_types=[
                # a
                TensorType(a_dtype, shape=(2, 4), device=device),
                # b
                TensorType(b_dtype, shape=(3, 4), device=device),
                # a_scales
                TensorType(a_scales_dtype, shape=(2, 1), device=device),
                # b_scales
                TensorType(b_scales_dtype, shape=(3, 1), device=device),
            ],
        )


class FusedQKVRaggedMatmulScaledFloat8:
    """Wrapper for testing fused_qkv_ragged_matmul_scaled_float8."""

    def __init__(
        self,
        kv_params: KVCacheParams,
        kv_collection: PagedKVCacheCollection,
        n_heads: int,
    ) -> None:
        self.kv_params = kv_params
        self.kv_collection = kv_collection
        self.n_heads = n_heads

    def __call__(
        self,
        input: TensorValue,
        input_row_offsets: TensorValue,
        wqkv: TensorValue,
        layer_idx: TensorValue,
        input_scale: TensorValue,
        weight_scale: TensorValue,
    ) -> TensorValue:
        return fused_qkv_ragged_matmul_scaled_float8(
            self.kv_params,
            input,
            input_row_offsets,
            wqkv,
            self.kv_collection,
            layer_idx,
            self.n_heads,
            input_scale,
            weight_scale,
        )


def test_fused_qkv_ragged_matmul_scaled_float8_valid():
    """Tests fused_qkv_ragged_matmul_scaled_float8 with all tensors on same device."""
    device = DeviceRef.CPU()

    # Create KV cache parameters
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    with Graph(
        "fused_qkv_ragged_matmul_scaled_float8",
        input_types=[
            # input
            TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # wqkv
            TensorType(DType.float8_e4m3fn, shape=(512, 1536), device=device),
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
            # input_scale
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # weight_scale
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # KV cache collection inputs
            # blocks: [num_pages, 2, n_kv_heads, page_size, head_dim]
            TensorType(
                DType.bfloat16, shape=(16, 2, 8, 128, 64), device=device
            ),
            # cache_lengths: [batch_size]
            TensorType(DType.uint32, shape=(2,), device=device),
            # lookup_table: [batch_size, max_pages]
            TensorType(DType.uint32, shape=(2, 8), device=device),
            # is_cache_empty: scalar
            TensorType(DType.uint32, shape=(), device=device),
        ],
    ) as graph:
        (
            input_tensor,
            input_row_offsets,
            wqkv,
            layer_idx,
            input_scale,
            weight_scale,
            blocks,
            cache_lengths,
            lookup_table,
            is_cache_empty,
        ) = [inp.tensor for inp in graph.inputs]

        # Create PagedKVCacheCollection
        kv_collection = PagedKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.paged",
                device=blocks.device,
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": kv_params.n_kv_heads_per_device,
                    "head_dim": kv_params.head_dim,
                    "page_size": 128,
                },
            )[0].opaque
        )

        # Now call the kernel - should not raise any errors when all devices match
        output = fused_qkv_ragged_matmul_scaled_float8(
            kv_params,
            input_tensor,
            input_row_offsets,
            wqkv,
            kv_collection,
            layer_idx,
            32,  # n_heads
            input_scale,
            weight_scale,
        )
        assert output.shape == [10, 32 * 64]  # [seq_len, n_heads * head_dim]
        assert output.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "input_dev, wqkv_dev, row_off_dev, in_scale_dev, w_scale_dev, err_msg_part",
    [
        # Individual device mismatches
        ("cpu", "gpu", "cpu", "cpu", "cpu", r"wqkv=gpu:0\n"),
        ("cpu", "cpu", "gpu", "cpu", "cpu", r"input_row_offsets=gpu:0\n"),
        ("cpu", "cpu", "cpu", "gpu", "cpu", r"input_scale=gpu:0\n"),
        ("cpu", "cpu", "cpu", "cpu", "gpu", r"weight_scale=gpu:0"),
        # Multiple device mismatches
        (
            "cpu",
            "gpu",
            "gpu",
            "cpu",
            "cpu",
            r"wqkv=gpu:0\n.*input_row_offsets=gpu:0\n",
        ),
        (
            "cpu",
            "cpu",
            "cpu",
            "gpu",
            "gpu",
            r"input_scale=gpu:0\n.*weight_scale=gpu:0",
        ),
        # All on wrong device
        (
            "cpu",
            "gpu",
            "gpu",
            "gpu",
            "gpu",
            r"wqkv=gpu:0\n.*input_row_offsets=gpu:0\n.*input_scale=gpu:0\n.*weight_scale=gpu:0",
        ),
        # Reverse case - input on GPU, others on CPU
        (
            "gpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
            r"wqkv=cpu:0\n.*input_row_offsets=cpu:0\n.*input_scale=cpu:0\n.*weight_scale=cpu:0",
        ),
    ],
)
def test_fused_qkv_ragged_matmul_scaled_float8_device_mismatch(
    input_dev: str,
    wqkv_dev: str,
    row_off_dev: str,
    in_scale_dev: str,
    w_scale_dev: str,
    err_msg_part: str,
) -> None:
    """Tests that device mismatches raise appropriate errors."""

    # Map device strings to DeviceRef
    def get_device(dev_str: str) -> DeviceRef:
        return DeviceRef.GPU(0) if dev_str == "gpu" else DeviceRef.CPU()

    # Create KV cache parameters (can use real object for device tests)
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    kv_collection = Mock(spec=PagedKVCacheCollection)

    with pytest.raises(ValueError, match=err_msg_part):
        Graph(
            "fused_qkv_ragged_matmul_scaled_float8",
            forward=FusedQKVRaggedMatmulScaledFloat8(
                kv_params, kv_collection, n_heads=32
            ),
            input_types=[
                # input
                TensorType(
                    DType.float8_e4m3fn,
                    shape=(10, 512),
                    device=get_device(input_dev),
                ),
                # input_row_offsets
                TensorType(
                    DType.uint32, shape=(3,), device=get_device(row_off_dev)
                ),
                # wqkv
                TensorType(
                    DType.float8_e4m3fn,
                    shape=(512, 1536),
                    device=get_device(wqkv_dev),
                ),
                # layer_idx - must always be on CPU
                TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
                # input_scale
                TensorType(
                    DType.bfloat16,
                    shape=(1, 1),
                    device=get_device(in_scale_dev),
                ),
                # weight_scale
                TensorType(
                    DType.bfloat16, shape=(1, 1), device=get_device(w_scale_dev)
                ),
            ],
        )


def test_fused_qkv_ragged_matmul_scaled_float8_layer_idx_device():
    """Tests that layer_idx must be on CPU device."""
    device = DeviceRef.GPU(0)

    # Create KV cache parameters
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    kv_collection = Mock(spec=PagedKVCacheCollection)

    with pytest.raises(
        ValueError,
        match="expected layer_idx to be on CPU device, but got gpu:0",
    ):
        Graph(
            "fused_qkv_ragged_matmul_scaled_float8",
            forward=FusedQKVRaggedMatmulScaledFloat8(
                kv_params, kv_collection, n_heads=32
            ),
            input_types=[
                # input
                TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
                # input_row_offsets
                TensorType(DType.uint32, shape=(3,), device=device),
                # wqkv
                TensorType(
                    DType.float8_e4m3fn, shape=(512, 1536), device=device
                ),
                # layer_idx - incorrectly on GPU
                TensorType(DType.uint32, shape=(), device=device),
                # input_scale
                TensorType(DType.bfloat16, shape=(1, 1), device=device),
                # weight_scale
                TensorType(DType.bfloat16, shape=(1, 1), device=device),
            ],
        )
