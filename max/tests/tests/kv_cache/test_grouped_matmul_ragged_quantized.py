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

"""Tests for grouped_matmul_ragged_quantized in max.nn.kernels."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.kernels import grouped_matmul_ragged_quantized


def _fp8_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _mxfp4_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.MXFP4,
        can_use_fused_mlp=False,
    )


def _nvfp4_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 8),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )


def test_grouped_matmul_ragged_quantized_fp8_valid() -> None:
    """Builds a graph for the FP8 grouped-MoE wrapper."""
    device = DeviceRef.CPU()
    with Graph(
        "grouped_matmul_ragged_quantized_fp8",
        input_types=[
            TensorType(DType.bfloat16, shape=(99, 512), device=device),
            TensorType(DType.float8_e4m3fn, shape=(3, 512, 256), device=device),
            TensorType(DType.float32, shape=(3, 4, 2), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
            TensorType(DType.int32, shape=(3,), device=device),
            TensorType(DType.uint32, shape=(2,), device=device),
        ],
    ) as graph:
        output = grouped_matmul_ragged_quantized(
            x=graph.inputs[0].tensor,
            weight=graph.inputs[1].tensor,
            weight_scale=graph.inputs[2].tensor,
            expert_start_indices=graph.inputs[3].tensor,
            expert_ids=graph.inputs[4].tensor,
            usage_stats=graph.inputs[5].tensor,
            quant_config=_fp8_config(),
        )

        assert output.shape == [99, 256]
        assert output.dtype == DType.bfloat16


def test_grouped_matmul_ragged_quantized_mxfp4_valid() -> None:
    """Builds a graph for the MXFP4 grouped-MoE wrapper."""
    device = DeviceRef.CPU()
    with Graph(
        "grouped_matmul_ragged_quantized_mxfp4",
        input_types=[
            TensorType(DType.bfloat16, shape=(99, 512), device=device),
            TensorType(DType.uint8, shape=(3, 256, 256), device=device),
            TensorType(DType.float8_e8m0fnu, shape=(3, 256, 16), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
            TensorType(DType.int32, shape=(3,), device=device),
            TensorType(DType.uint32, shape=(2,), device=device),
        ],
    ) as graph:
        output = grouped_matmul_ragged_quantized(
            x=graph.inputs[0].tensor,
            weight=graph.inputs[1].tensor,
            weight_scale=graph.inputs[2].tensor,
            expert_start_indices=graph.inputs[3].tensor,
            expert_ids=graph.inputs[4].tensor,
            usage_stats=graph.inputs[5].tensor,
            quant_config=_mxfp4_config(),
        )

        assert output.shape == [99, 256]
        assert output.dtype == DType.bfloat16


def test_grouped_matmul_ragged_quantized_unsupported_format() -> None:
    """Rejects quantization formats that do not yet have a grouped MoE path."""
    device = DeviceRef.CPU()
    with Graph(
        "grouped_matmul_ragged_quantized_unsupported",
        input_types=[
            TensorType(DType.bfloat16, shape=(99, 512), device=device),
            TensorType(DType.uint8, shape=(3, 256, 256), device=device),
            TensorType(DType.float8_e4m3fn, shape=(3, 256, 32), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
            TensorType(DType.int32, shape=(3,), device=device),
            TensorType(DType.uint32, shape=(2,), device=device),
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match="Unsupported quantization format for grouped matmul",
        ):
            grouped_matmul_ragged_quantized(
                x=graph.inputs[0].tensor,
                weight=graph.inputs[1].tensor,
                weight_scale=graph.inputs[2].tensor,
                expert_start_indices=graph.inputs[3].tensor,
                expert_ids=graph.inputs[4].tensor,
                usage_stats=graph.inputs[5].tensor,
                quant_config=_nvfp4_config(),
            )
