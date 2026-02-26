# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for FP4 matmul kernels in max.nn.kernels."""

from __future__ import annotations

from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul_fp4,
    fused_qkv_ragged_matmul_scaled_float4,
    grouped_dynamic_scaled_nvfp4_matmul,
    quantize_dynamic_block_scaled_fp4,
)
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)


class FusedQKVRaggedMatmulScaledFloat4:
    """Wrapper for testing fused_qkv_ragged_matmul_scaled_float4."""

    def __init__(
        self,
        kv_params: KVCacheParams,
        kv_collection: PagedCacheValues,
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
        kv_scales: TensorValue,
        weight_scale: TensorValue,
        weight_scale_2: TensorValue,
    ) -> TensorValue:
        x, x_scales = quantize_dynamic_block_scaled_fp4(
            input,
            tensor_sf=1.0 / input_scale,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,  # fp4-e2m1fnX2
        )

        weight_scale = weight_scale.to(x.device)
        weight_scale = block_scales_interleave(
            weight_scale,
        )

        return fused_qkv_ragged_matmul_scaled_float4(
            self.kv_params,
            x,
            input_row_offsets,
            wqkv,
            self.kv_collection,
            layer_idx,
            self.n_heads,
            x_scales,
            weight_scale,
            input_scale * weight_scale_2,
            kv_scales,
        )


def test_fused_qkv_ragged_matmul_scaled_float4_valid() -> None:
    """Tests fused_qkv_ragged_matmul_scaled_float4 with all tensors on same device."""
    device = DeviceRef.CPU()

    # Create KV cache parameters
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        cache_strategy="paged",
        page_size=128,
        devices=[device],
    )

    with Graph(
        "fused_qkv_ragged_matmul_scaled_float4",
        input_types=[
            # input
            TensorType(DType.bfloat16, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # wqkv
            TensorType(DType.uint8, shape=(512, 1536), device=device),
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
            # input_scale
            TensorType(DType.float32, shape=(), device=device),
            # kv_scales
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # weight_scale
            TensorType(
                DType.float8_e4m3fn, shape=(1536, 512 // 16), device=device
            ),
            # weight_scale_2
            TensorType(DType.float32, shape=(), device=device),
            # KV cache collection inputs
            # blocks: [num_pages, 2, n_kv_heads, page_size, head_dim]
            BufferType(
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
            kv_scales,
            weight_scale,
            weight_scale_2,
            blocks,
            cache_lengths,
            lookup_table,
            is_cache_empty,
        ) = graph.inputs

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            is_cache_empty.tensor,
        )

        tester = FusedQKVRaggedMatmulScaledFloat4(kv_params, kv_collection, 32)

        # Now call the kernel - should not raise any errors when all devices match
        output = tester(
            input_tensor.tensor,
            input_row_offsets.tensor,
            wqkv.tensor,
            layer_idx.tensor,
            input_scale.tensor,
            kv_scales.tensor,
            weight_scale.tensor,
            weight_scale_2.tensor,
        )
        assert output.shape == [10, 32 * 64]  # [seq_len, n_heads * head_dim]
        assert output.dtype == DType.bfloat16


def test_dynamic_block_scaled_1d1d_matmul_fp4() -> None:
    """Tests dynamic_block_scaled_1d1d_matmul_fp4 with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "dynamic_block_scaled_matmul_fp4",
        input_types=[
            # a
            TensorType(DType.uint8, shape=(127, 129), device=device),
            # b
            TensorType(DType.uint8, shape=(129, 129), device=device),
            # a_scales
            TensorType(
                DType.float8_e4m3fn, shape=(1, 5, 32, 4, 4), device=device
            ),
            # b_scales
            TensorType(
                DType.float8_e4m3fn, shape=(2, 5, 32, 4, 4), device=device
            ),
        ],
    ) as graph:
        a, b, a_scales, b_scales = (inp.tensor for inp in graph.inputs)

        output = dynamic_block_scaled_matmul_fp4(
            a,
            b,
            a_scales,
            b_scales,
            1.0,
        )
        assert output.shape == [127, 129]
        assert output.dtype == DType.bfloat16


def test_quantize_dynamic_block_scaled_fp4() -> None:
    """Tests quantize_dynamic_block_scaled_fp4 with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "quantize_dynamic_block_scaled_fp4",
        input_types=[
            # input
            TensorType(DType.bfloat16, shape=(129, 136), device=device),
        ],
    ) as graph:
        (input,) = (inp.tensor for inp in graph.inputs)

        quantized_output, scales = quantize_dynamic_block_scaled_fp4(
            input,
            1.0,
        )
        assert quantized_output.shape == [129, 68]
        assert quantized_output.dtype == DType.uint8
        assert scales.shape == [2, 3, 32, 4, 4]
        assert scales.dtype == DType.float8_e4m3fn


def test_block_scales_interleave() -> None:
    """Tests block_scales_interleave with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "block_scales_interleave",
        input_types=[
            # scales
            TensorType(DType.float8_e4m3fn, shape=(129, 136), device=device),
        ],
    ) as graph:
        (scales,) = (inp.tensor for inp in graph.inputs)

        scales_interleaved = block_scales_interleave(
            scales,
        )
        assert scales_interleaved.shape == [2, 34, 32, 4, 4]
        assert scales_interleaved.dtype == DType.float8_e4m3fn


# NVFP4 scale factor parameters (constants).
# NOTE: tcgen05 scale factors are stored in a 5D layout:
# (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4).
# Shape of scale factor MN-group.
_SF_ATOM_M = (32, 4)
# Number of scale factors per K-group.
_SF_ATOM_K = 4
# Number of elements per scale factor.
_SF_VECTOR_SIZE = 16
_SF_MN_GROUP_SIZE = _SF_ATOM_M[0] * _SF_ATOM_M[1]  # 128
_SF_K_GROUP_SIZE = _SF_ATOM_K * _SF_VECTOR_SIZE  # 64


def _get_nvfp4_input_types(
    device: DeviceRef,
    num_experts: int = 3,
    total_tokens: int = 99,
    N: int = 256,
    K: int = 512,
    hidden_dtype: DType = DType.uint8,
    a_scales_dtype: DType = DType.float8_e4m3fn,
    a_scales_shape: tuple[int, ...] | None = None,
) -> list[TensorType | BufferType]:
    """Returns input types for grouped_dynamic_scaled_nvfp4_matmul tests."""
    num_scale_rows = (total_tokens + _SF_MN_GROUP_SIZE - 1) // _SF_MN_GROUP_SIZE
    K_groups = (K + _SF_K_GROUP_SIZE - 1) // _SF_K_GROUP_SIZE
    N_groups = (N + _SF_MN_GROUP_SIZE - 1) // _SF_MN_GROUP_SIZE

    if a_scales_shape is None:
        a_scales_shape = (
            num_scale_rows,
            K_groups,
            _SF_ATOM_M[0],
            _SF_ATOM_M[1],
            _SF_ATOM_K,
        )

    return [
        TensorType(hidden_dtype, shape=(total_tokens, K // 2), device=device),
        TensorType(DType.uint8, shape=(num_experts, N, K // 2), device=device),
        TensorType(a_scales_dtype, shape=a_scales_shape, device=device),
        TensorType(
            DType.float8_e4m3fn,
            shape=(
                num_experts,
                N_groups,
                K_groups,
                _SF_ATOM_M[0],
                _SF_ATOM_M[1],
                _SF_ATOM_K,
            ),
            device=device,
        ),
        TensorType(
            DType.uint32, shape=(1,), device=device
        ),  # expert_start_indices
        TensorType(
            DType.uint32, shape=(num_experts,), device=device
        ),  # a_scale_offsets
        TensorType(
            DType.int32, shape=(num_experts,), device=device
        ),  # expert_ids
        TensorType(
            DType.float32, shape=(num_experts,), device=device
        ),  # expert_scales
        TensorType(
            DType.uint32, shape=(2,), device=device
        ),  # expert_usage_stats_host
    ]


def _call_nvfp4_matmul(
    input_types: list[TensorType | BufferType],
) -> TensorValue:
    """Builds a graph calling grouped_dynamic_scaled_nvfp4_matmul."""
    with Graph("test_nvfp4", input_types=input_types) as graph:
        inputs = graph.inputs
        return grouped_dynamic_scaled_nvfp4_matmul(
            inputs[0].tensor,
            inputs[1].tensor,
            inputs[2].tensor,
            inputs[3].tensor,
            inputs[4].tensor,
            inputs[5].tensor,
            inputs[6].tensor,
            inputs[7].tensor,
            inputs[8].tensor,
        )


def test_grouped_dynamic_scaled_nvfp4_matmul_valid() -> None:
    """Tests grouped_dynamic_scaled_nvfp4_matmul with valid inputs."""
    input_types = _get_nvfp4_input_types(DeviceRef.CPU())
    output = _call_nvfp4_matmul(input_types)
    assert output.shape == [99, 256]
    assert output.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    [
        (
            {"hidden_dtype": DType.bfloat16},
            TypeError,
            "hidden_states and weight dtypes must be uint8 for NVFP4",
        ),
        (
            {"a_scales_dtype": DType.float32},
            TypeError,
            "a_scales and b_scales dtypes must be float8_e4m3fn for NVFP4",
        ),
        (
            {"a_scales_shape": (1, 8)},
            ValueError,
            "expected a_scales of rank 5 and b_scales of rank 6",
        ),
    ],
)
def test_grouped_dynamic_scaled_nvfp4_matmul_invalid(
    kwargs: dict[str, Any], error_type: type[Exception], error_match: str
) -> None:
    """Tests grouped_dynamic_scaled_nvfp4_matmul rejects invalid inputs."""
    input_types = _get_nvfp4_input_types(DeviceRef.CPU(), **kwargs)
    with pytest.raises(error_type, match=error_match):
        _call_nvfp4_matmul(input_types)
