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

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul_fp4,
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
        assert scales_interleaved.shape == [2, 3, 32, 4, 4]
        assert scales_interleaved.dtype == DType.float8_e4m3fn
