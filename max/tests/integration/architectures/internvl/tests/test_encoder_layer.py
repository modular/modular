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

"""Tests for InternVL encoder layer."""

import os

import pytest
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVLConfig,
)
from internvl_impl.modeling_intern_vit import (
    InternVisionEncoderLayer as HFInternVisionEncoderLayer,
)
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Signals
from max.pipelines.architectures.internvl.internvl import (
    InternVisionEncoderLayer,
)
from max.pipelines.architectures.internvl.model_config import InternVLConfig
from torch.utils.dlpack import from_dlpack
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_torch_outputs(
    hf_config: HFInternVLConfig,
    input_tensor: torch.Tensor,
    encoder_layer_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL implementation."""
    # Create the HuggingFace encoder layer
    layer = (
        HFInternVisionEncoderLayer(
            hf_config, drop_path_rate=hf_config.drop_path_rate
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights using state_dict, handling QKV weight splitting
    layer.load_state_dict(encoder_layer_weights, strict=True)
    layer.eval()

    # Forward pass
    outputs = layer(input_tensor)
    output = outputs[0] if isinstance(outputs, tuple) else outputs

    return output


def build_max_encoder_layer(
    max_config: InternVLConfig,
    vision_encoder_layer_weights: dict[str, torch.Tensor],
    input_shape: tuple[int, int, int],
    dtype: DType,
) -> tuple[Model, Signals]:
    """Compiles the MAX InternVisionEncoderLayer graph.

    Nothing here touches the device, so `precompile_mefs` can record the
    compile on a CPU worker.
    """
    max_weights = convert_hf_to_max_weights(vision_encoder_layer_weights)

    encoder_layer = InternVisionEncoderLayer(max_config)
    encoder_layer.load_state_dict(
        state_dict=max_weights,
        strict=True,
    )

    session = InferenceSession(devices=[Accelerator(0)])

    device_ref = DeviceRef.GPU()
    input_type = TensorType(dtype, input_shape, device=device_ref)
    signals = Signals(devices=[device_ref])

    with Graph(
        "InternVisionEncoderLayer",
        input_types=(input_type, *signals.input_types()),
    ) as graph:
        x = graph.inputs[0]
        signal_buffers = [v.buffer for v in graph.inputs[1:]]
        outputs = encoder_layer([x.tensor], signal_buffers)
        graph.output(outputs[0])

    compiled = session.load(graph, weights_registry=encoder_layer.state_dict())
    return compiled, signals


def execute_max_encoder_layer(
    compiled: Model,
    signals: Signals,
    input_tensor: torch.Tensor,
    device: Device,
) -> torch.Tensor:
    """Runs the compiled MAX InternVisionEncoderLayer on `input_tensor`."""
    result = compiled.execute(
        Buffer.from_dlpack(input_tensor).to(device), *signals.buffers()
    )
    return from_dlpack(result[0])


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
def test_vision_encoder_layer(config_name: ConfigNames) -> None:
    """Test InternVisionEncoderLayer against PyTorch reference."""
    torch.manual_seed(42)

    # Load HuggingFace config and generate weights
    hf_config = get_config_loader().load_hf_vision_config(config_name)
    internvl_config = get_config_loader().create_internvl_config(config_name)
    vision_encoder_layer_weights = get_weight_generator(
        config_name
    ).generate_vision_encoder_layer_weights()

    batch_size = 1
    seq_len = 257  # 256 patches + 1 CLS token
    hidden_size = internvl_config.vision_config.hidden_size

    # Compile before anything touches the GPU, so `precompile_mefs` records
    # the compile instead of stopping at the first device allocation.
    compiled, signals = build_max_encoder_layer(
        max_config=internvl_config,
        vision_encoder_layer_weights=vision_encoder_layer_weights,
        input_shape=(batch_size, seq_len, hidden_size),
        dtype=DType.bfloat16,
    )

    vision_input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch.bfloat16
    ).cuda()

    torch_output = generate_torch_outputs(
        hf_config,
        vision_input_tensor,
        vision_encoder_layer_weights,
    )

    max_output = execute_max_encoder_layer(
        compiled, signals, vision_input_tensor, Accelerator()
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision encoder layer outputs do not match",
    )
