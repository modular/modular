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

"""Tests for InternVL vision embeddings multi-GPU execution."""

import os

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


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
def test_vision_embeddings_multi_gpu_execution(
    config_name: ConfigNames,
) -> None:
    """Test InternVisionEmbeddings execution on 2 GPUs with large image."""
    # Set seed for deterministic results
    torch.manual_seed(42)

    n_devices = 2
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Load HuggingFace config and generate weights
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    height, width = 896, 672
    batch_size = 1  # Define batch_size explicitly

    # Create test inputs
    pixel_values = torch.randn(
        batch_size, 3, height, width, dtype=torch.float32
    ).to("cuda")

    # Convert to patches for the test
    pixel_values_np = (
        pixel_values.permute(0, 2, 3, 1).cpu().numpy()
    )  # NCHW -> NHWC

    # Split each image in the batch into patches.
    all_patches = []
    for i in range(batch_size):
        patches = extract_patches_from_image(
            pixel_values_np[i],
            patch_size=internvl_config.vision_config.patch_size,
        )
        all_patches.append(patches)
    patches = np.stack(all_patches).astype(np.float32)

    # Shape is now (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    input_shape = patches.shape

    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Build graph with f32 input
    input_type_gpu0 = TensorType(
        DType.float32,  # f32 input
        shape=input_shape,
        device=DeviceRef.GPU(0),
    )

    # Create embeddings for each GPU
    embeddings_gpu0 = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(0))
    embeddings_gpu0.load_state_dict(
        state_dict=max_weights,
    )

    embeddings_gpu1 = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(1))
    embeddings_gpu1.load_state_dict(
        state_dict=max_weights,
    )

    # Build separate graphs for each GPU
    graph0 = Graph(
        "InternVisionEmbeddings_GPU0",
        forward=embeddings_gpu0,
        input_types=(input_type_gpu0,),
    )

    input_type_gpu1 = TensorType(
        DType.float32,
        shape=input_shape,
        device=DeviceRef.GPU(1),
    )

    graph1 = Graph(
        "InternVisionEmbeddings_GPU1",
        forward=embeddings_gpu1,
        input_types=(input_type_gpu1,),
    )

    # Create sessions and compile
    session0 = InferenceSession(devices=[Accelerator(0)])
    session1 = InferenceSession(devices=[Accelerator(1)])

    compiled0 = session0.load(graph0, weights_registry=max_weights)
    compiled1 = session1.load(graph1, weights_registry=max_weights)

    # Convert patches to torch tensor for execution.
    patches_tensor = torch.from_numpy(patches).to(torch.float32)

    # Execute on GPU 0
    patches_gpu0 = patches_tensor.to("cuda:0")
    input_tensor0 = Tensor.from_dlpack(patches_gpu0.contiguous()).to(
        Accelerator(0)
    )
    result0 = compiled0.execute(input_tensor0)[0]
    assert isinstance(result0, Tensor)
    result0 = result0.to(CPU())

    # Copy input to GPU 1 and execute
    patches_gpu1 = patches_tensor.to("cuda:1")
    with torch.cuda.device(1):
        input_tensor1 = Tensor.from_dlpack(patches_gpu1.contiguous()).to(
            Accelerator(1)
        )
    result1 = compiled1.execute(input_tensor1)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to(CPU())

    output0 = from_dlpack(result0)
    output1 = from_dlpack(result1)

    # Verify output shape
    expected_num_patches = (
        height // internvl_config.vision_config.patch_size
    ) * (width // internvl_config.vision_config.patch_size)
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        internvl_config.vision_config.hidden_size,
    )

    assert output0.shape == expected_shape
    assert output1.shape == expected_shape

    # Compare outputs from both GPUs.
    assert_tensors_close(
        output0,
        output1,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision embeddings multi-GPU outputs do not match",
    )

    print(
        f"Successfully tested execution on 2 GPUs with {height}x{width} image"
    )
