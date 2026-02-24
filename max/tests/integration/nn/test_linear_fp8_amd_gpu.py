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
"""Test Linear layer with FP8 quantization on AMD GPUs."""

import torch
from max.driver import CPU
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph
from max.nn import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Linear,
)
from max.nn.float8_config import Float8WeightScaleSpec


def test_linear_fp8_amd_conversion_static_scale(
    gpu_session: InferenceSession,
) -> None:
    """Test Linear layer applies AMD FP8 conversion when needed."""

    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
            origin=Float8ScaleOrigin.STATIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.TENSOR,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    weights = [1.0, 2.0, -1.0, 0.0, -0.0]
    input = Tensor.constant([weights], dtype=DType.bfloat16)

    fp8_graph = Graph(
        "linear_test",
        Linear(
            name="linear",
            in_dim=5,
            out_dim=1,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(),
            float8_config=float8_config,
        ),
        input_types=[input.type],
    )

    fp8_model = gpu_session.load(
        fp8_graph,
        weights_registry={
            "linear.weight_scale": Tensor.constant([1.0], device=CPU()),
            "linear.input_scale": Tensor.constant([1.0], device=CPU()),
            "linear.weight": Tensor.constant(
                weights,
                dtype=DType.float8_e4m3fn,
                device=CPU(),
            ),
        },
    )

    bf16_graph = Graph(
        "linear_test_bf16",
        Linear(
            name="linear",
            in_dim=5,
            out_dim=1,
            dtype=DType.bfloat16,
            device=DeviceRef.GPU(),
        ),
        input_types=[input.type],
    )

    bf16_model = gpu_session.load(
        bf16_graph,
        weights_registry={
            "linear.weight": Tensor.constant(
                weights, dtype=DType.bfloat16, device=CPU()
            ),
        },
    )

    fp8_model = F.functional(fp8_model)  # type: ignore
    bf16_model = F.functional(bf16_model)  # type: ignore

    fp8_result = fp8_model(input)[0]
    bf16_result = bf16_model(input)[0]

    assert isinstance(fp8_result, Tensor)
    assert isinstance(bf16_result, Tensor)

    assert torch.isfinite(torch.from_dlpack(fp8_result)).all(), (
        "FP8 result should be finite (no NaN or Inf)"
    )
    assert fp8_result.item() == bf16_result.item(), (
        "FP8 and BF16 outputs should be similar"
    )


def test_linear_fp8_amd_conversion_dynamic_scale(
    gpu_session: InferenceSession,
) -> None:
    """Test Linear layer applies AMD FP8 conversion with dynamic scaling."""

    float8_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.COLWISE,
            origin=Float8ScaleOrigin.DYNAMIC,
        ),
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.ROWWISE,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    weight_scale = Tensor.constant([1.0], dtype=DType.float8_e4m3fn)
    base_values = Tensor.constant(
        [[1.0, 2.0, -1.0, 0.0, -0.0]], dtype=DType.float8_e4m3fn
    )
    weights = F.tile(base_values, (16, 7))[:16, :32]

    inputs = random.uniform((4, 32), range=(0, 0.5))

    graph = Graph(
        "linear_test",
        Linear(
            name="linear",
            in_dim=32,
            out_dim=16,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(),
            float8_config=float8_config,
        ),
        input_types=[inputs.type],
    )

    model = gpu_session.load(
        graph,
        weights_registry={
            "linear.weight_scale": weight_scale.to(CPU()),
            "linear.weight": weights.to(CPU()),
        },
    )
    fp8_result = model.execute(inputs)[0]

    # TODO: The results of dynamic quantized matmul are quite off
    assert torch.isfinite(torch.from_dlpack(fp8_result)).all(), (
        "Dynamic scaling: FP8 result should be finite (no NaN or Inf)"
    )
