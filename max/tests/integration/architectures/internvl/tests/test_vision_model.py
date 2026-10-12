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

"""Tests for InternVL vision model."""

import os

import numpy as np
import pytest
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVLConfig,
)
from internvl_impl.modeling_intern_vit import InternVLVisionModelWithProjection
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Signals
from max.pipelines.architectures.internvl.internvl import InternVLVisionModel
from max.pipelines.architectures.internvl.model_config import InternVLConfig
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    hf_config: HFInternVLConfig,
    vision_model_weights: dict[str, torch.Tensor],
    internvl_config: InternVLConfig,
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL implementation with projection."""
    # Create the InternVL Vision Model with projection
    model = (
        InternVLVisionModelWithProjection(
            vision_config=hf_config,
            llm_hidden_size=internvl_config.llm_config.hidden_size,
            downsample_ratio=internvl_config.downsample_ratio,
        )
        .to(torch.bfloat16)
        .to("cuda")
        .eval()
    )

    model.load_state_dict(vision_model_weights, strict=True)
    model.eval()

    # Convert from NHWC to NCHW for PyTorch
    pixel_values_nchw = pixel_values.permute(0, 3, 1, 2)

    # Get projected vision features
    vision_features = model(pixel_values=pixel_values_nchw, select_layer=-1)

    # Flatten to match MAX output format
    # [batch, seq_len, hidden_dim] -> [batch * seq_len, hidden_dim]
    batch_size = vision_features.shape[0]
    seq_len = vision_features.shape[1]
    vision_features = vision_features.reshape(batch_size * seq_len, -1)

    return vision_features


def generate_max_outputs(
    pixel_values: torch.Tensor,
    intern_vl_config: InternVLConfig,
    vision_model_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVLVisionModel implementation."""

    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()

    # Convert HuggingFace weights to MAX format using weight adapters
    max_weights = convert_hf_to_max_weights(vision_model_weights)

    # Create the vision model
    vision_model = InternVLVisionModel(intern_vl_config)

    # Load all weights with proper scoping - now strict=True should work
    vision_model.load_state_dict(max_weights, strict=True)

    session = InferenceSession(devices=[device])

    # Extract shape information: pixel_values is NHWC here.
    batch_size = pixel_values.shape[0]

    # Convert to float32 first since numpy doesn't support bfloat16.
    pixel_values_np = pixel_values.cpu().float().numpy()

    # Split each image in the batch into patches.
    all_patches = []
    for i in range(batch_size):
        img = pixel_values_np[i]
        patches = extract_patches_from_image(
            img, patch_size=intern_vl_config.vision_config.patch_size
        )
        all_patches.append(patches)

    # Stack all patches - shape: (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    pixel_values_np = np.stack(all_patches)

    # Convert back to torch tensor and move to device
    pixel_values_patched = (
        torch.from_numpy(pixel_values_np).to(dtype=torch.bfloat16)
    ).contiguous()

    if is_gpu:
        pixel_values_patched = pixel_values_patched.cuda()
    else:
        pixel_values_patched = pixel_values_patched.cpu()

    # Build the graph
    input_shape = pixel_values_patched.shape
    input_type = TensorType(dtype, input_shape, device=device_ref)

    # Create signal types for distributed communication.
    signals = Signals(devices=[device_ref])

    with Graph(
        "InternVLVisionModel", input_types=(input_type, *signals.input_types())
    ) as graph:
        x = graph.inputs[0]
        signal_args = graph.inputs[1:]

        # Extract signal buffers (one per device).
        signal_buffers = [v.buffer for v in signal_args]

        outputs = vision_model([x.tensor], signal_buffers)
        graph.output(outputs[0])

    compiled = session.load(graph, weights_registry=vision_model.state_dict())

    # Execute the model and get the first result
    # Create signal buffer tensors for execution.
    signal_buffer_tensors = signals.buffers()

    result = compiled.execute(
        Buffer.from_dlpack(pixel_values_patched).to(device),
        *signal_buffer_tensors,
    )
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


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
def test_vision_model(config_name: ConfigNames) -> None:
    """Test complete InternVLVisionModel against PyTorch reference."""
    torch.manual_seed(42)

    # Create test instance
    hf_config = get_config_loader().load_hf_vision_config(config_name)
    internvl_config = get_config_loader().create_internvl_config(config_name)
    vision_model_weights = get_weight_generator(
        config_name
    ).generate_vision_model_weights()

    # Create test input tensor
    batch_size = 1
    height = width = internvl_config.vision_config.image_size  # 448
    channels = 3

    vision_pixel_values = torch.randn(
        batch_size, height, width, channels, dtype=torch.bfloat16
    ).cuda()

    # Generate reference output
    torch_output = generate_torch_outputs(
        vision_pixel_values,
        hf_config,
        vision_model_weights,
        internvl_config,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        vision_pixel_values,
        internvl_config,
        vision_model_weights,
        DType.bfloat16,
        Accelerator(0),
    )

    # TODO: Errors are far too high (they're small on the smaller variants even though they do worse on the e2e tests???)
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=0.25,
        atol=0.85,
        message="Vision model outputs do not match",
    )
