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
"""Test MLP layer with NVFP4 quantization.

Tests the MLP layer with NVFP4 (modelopt) quantization config matching
nvidia/Llama-3.1-8B-Instruct-NVFP4:
- Input scaling: Static, block-wise (1, 16)
- Weight scaling: Static, block-wise (1, 8)
- Quantization method: modelopt NVFP4
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, TensorValue
from max.graph.weights import WeightData
from max.nn import MLP
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200


def _skip_if_not_supported() -> None:
    """Skip test if hardware doesn't support FP4."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available for FP4 MLP test")
    if accelerator_api() == "hip":
        pytest.skip("FP4 kernel only supports Nvidia GPUs")
    if not is_b100_b200():
        pytest.skip("FP4 kernel requires B100 or B200")


def _create_nvfp4_config() -> QuantConfig:
    """Creates QuantConfig matching nvidia/Llama-3.1-8B-Instruct-NVFP4.

    NVFP4 format uses:
    - Static, block-wise (1, 16) input scaling
    - Static, block-wise (1, 8) weight scaling

    Returns:
        QuantConfig for NVFP4 quantization.
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16),
    )
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers={0},
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.NVFP4,
    )


def _generate_fp4_weights(
    out_dim: int,
    in_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates NVFP4 weights with scales.

    Args:
        out_dim: Output dimension.
        in_dim: Input dimension (must be divisible by 2).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (weight, weight_scale, weight_scale_2, input_scale).
    """
    torch.manual_seed(seed)

    packed_in_dim = in_dim // 2
    scale_k_dim = packed_in_dim // 8

    weight = torch.randint(0, 256, (out_dim, packed_in_dim), dtype=torch.uint8)

    fp4_scale_min = 50.0
    fp4_scale_max = 150.0
    weight_scale = (
        torch.rand(out_dim, scale_k_dim, dtype=torch.float32)
        * (fp4_scale_max - fp4_scale_min)
        + fp4_scale_min
    ).to(torch.float8_e4m3fn)

    weight_scale_2 = torch.rand((), dtype=torch.float32) * 1e-4
    input_scale = torch.rand((), dtype=torch.float32) * 1e-3

    return weight, weight_scale, weight_scale_2, input_scale


def _wrap_fp8_tensor(tensor: torch.Tensor, name: str) -> WeightData:
    """Wraps an FP8 tensor as WeightData for dlpack compatibility."""
    return WeightData(
        Buffer.from_dlpack(tensor.cpu().view(torch.uint8)).view(
            DType.float8_e4m3fn
        ),
        name,
        DType.float8_e4m3fn,
        Shape(tensor.shape),
    )


def _create_fp4_mlp_state_dict(
    gate_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_weight_scale_2: torch.Tensor,
    gate_input_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_scale_2: torch.Tensor,
    down_input_scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_scale_2: torch.Tensor,
    up_input_scale: torch.Tensor,
) -> dict[str, WeightData | torch.Tensor]:
    """Creates state dict for MAX MLP layer with FP4 weights.

    Returns:
        State dict mapping weight names to WeightData or tensors.
    """
    return {
        "gate_proj.weight": gate_weight.cpu(),
        "gate_proj.weight_scale": _wrap_fp8_tensor(
            gate_weight_scale, "gate_proj.weight_scale"
        ),
        "gate_proj.weight_scale_2": gate_weight_scale_2.cpu(),
        "gate_proj.input_scale": gate_input_scale.cpu(),
        "down_proj.weight": down_weight.cpu(),
        "down_proj.weight_scale": _wrap_fp8_tensor(
            down_weight_scale, "down_proj.weight_scale"
        ),
        "down_proj.weight_scale_2": down_weight_scale_2.cpu(),
        "down_proj.input_scale": down_input_scale.cpu(),
        "up_proj.weight": up_weight.cpu(),
        "up_proj.weight_scale": _wrap_fp8_tensor(
            up_weight_scale, "up_proj.weight_scale"
        ),
        "up_proj.weight_scale_2": up_weight_scale_2.cpu(),
        "up_proj.input_scale": up_input_scale.cpu(),
    }


def test_mlp_fp4_nvfp4() -> None:
    """Tests MLP layer with NVFP4 quantization config."""
    _skip_if_not_supported()
    hidden_dim = 256
    feed_forward_length = 512

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)
    fp4_config = _create_nvfp4_config()

    gate_weight, gate_scale, gate_scale_2, gate_input = _generate_fp4_weights(
        feed_forward_length, hidden_dim, seed=42
    )
    down_weight, down_scale, down_scale_2, down_input = _generate_fp4_weights(
        hidden_dim, feed_forward_length, seed=43
    )
    up_weight, up_scale, up_scale_2, up_input = _generate_fp4_weights(
        feed_forward_length, hidden_dim, seed=44
    )

    state_dict = _create_fp4_mlp_state_dict(
        gate_weight,
        gate_scale,
        gate_scale_2,
        gate_input,
        down_weight,
        down_scale,
        down_scale_2,
        down_input,
        up_weight,
        up_scale,
        up_scale_2,
        up_input,
    )

    mlp = MLP(
        dtype=DType.uint8,
        quantization_encoding=None,
        hidden_dim=hidden_dim,
        feed_forward_length=feed_forward_length,
        devices=[device_ref],
        activation_function="silu",
        has_bias=False,
        quant_config=fp4_config,
    )
    mlp.load_state_dict(state_dict)

    seq_len = 4
    x = torch.randn((seq_len, hidden_dim), dtype=torch.bfloat16) * 0.1
    x = x.cuda()

    session = InferenceSession(devices=[device])
    with Graph(
        "MLP_FP4_Test",
        input_types=[
            TensorType(
                DType.bfloat16,
                (seq_len, hidden_dim),
                device=device_ref,
            ),
        ],
    ) as graph:
        (graph_input,) = graph.inputs
        assert isinstance(graph_input, TensorValue)
        graph_output = mlp(graph_input)
        graph.output(graph_output)

    compiled = session.load(graph, weights_registry=mlp.state_dict())
    max_output = compiled.execute(x)[0]

    assert isinstance(max_output, Buffer)
    max_result = torch.from_dlpack(max_output)

    assert max_result.shape == (seq_len, hidden_dim)
    assert torch.isfinite(max_result).all(), "MAX output contains NaN or Inf"
