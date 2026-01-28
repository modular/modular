# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Tests for FP8 matmul kernels in max.nn.kernels."""

from __future__ import annotations

from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.legacy.kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul_fp4,
    grouped_dynamic_scaled_nvfp4_matmul,
    quantize_dynamic_block_scaled_fp4,
)


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
        TensorType(DType.uint32, shape=(1,), device=device),
        TensorType(DType.int32, shape=(num_experts,), device=device),
        TensorType(DType.float32, shape=(num_experts,), device=device),
        TensorType(DType.uint32, shape=(2,), device=device),
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
