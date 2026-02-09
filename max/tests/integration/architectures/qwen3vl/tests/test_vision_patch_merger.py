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

"""Tests for Qwen3VL vision position patch embeddings."""

from typing import Any

import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    VisionPatchMerger,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeVisionConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionPatchMerger as HFQwen3VLMoeVisionPatchMerger,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import load_weights_to_hf_merger
from utils.weight_generator import get_weight_generator

RTOL = 8e-2
ATOL = 8e-2


@pytest.fixture
def config_loader() -> Any:
    """Get config loader instance."""
    return get_config_loader()


@pytest.fixture
def weight_generator(config_loader: Any) -> Any:
    """Get weight generator instance."""
    return get_weight_generator(ConfigNames.QWEN3VL_30B)


@pytest.fixture
def vision_config(config_loader: Any) -> dict[str, Any]:
    """Get vision config."""
    config = config_loader.load_config(ConfigNames.QWEN3VL_30B)
    return config["vision_config"]


@torch.no_grad()
def generate_torch_outputs(
    input_tensor: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vision_config: dict[str, Any],
    use_postshuffle_norm: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate reference outputs using the HuggingFace implementation."""
    # Create the reference model
    hf_config = Qwen3VLMoeVisionConfig()

    hf_merger = HFQwen3VLMoeVisionPatchMerger(
        config=hf_config, use_postshuffle_norm=use_postshuffle_norm
    )
    hf_merger = hf_merger.to(device=input_tensor.device, dtype=dtype)
    hf_merger.eval()

    # Load weights
    load_weights_to_hf_merger(hf_merger, weights)

    # Forward pass
    with torch.no_grad():
        output = hf_merger(input_tensor)

    return output


def generate_max_outputs(
    input_tensor: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vision_config: dict[str, Any],
    use_postshuffle_norm: bool,
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX VisionPatchMerger implementation."""

    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()

    # Create MAX patch merger
    max_merger = VisionPatchMerger(
        dtype=dtype,
        devices=[device_ref],
        hidden_size=vision_config["hidden_size"],
        spatial_merge_size=vision_config["spatial_merge_size"],
        out_hidden_size=vision_config["out_hidden_size"],
        use_postshuffle_norm=use_postshuffle_norm,
    )

    # Load weights
    max_merger.load_state_dict(weights, strict=True)

    session = InferenceSession(devices=[device])

    # Define input type
    input_type = TensorType(dtype, shape=input_tensor.shape, device=device_ref)

    with Graph(
        "VisionPatchMerger",
        input_types=(input_type,),
    ) as graph:
        x = graph.inputs[0]
        output = max_merger(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=max_merger.state_dict())

    # Execute the model
    result = compiled.execute(
        Buffer.from_dlpack(input_tensor).to(device),
    )
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "use_postshuffle_norm",
    [False, True],
)
def test_vision_patch_merger(
    vision_config: dict[str, Any],
    weight_generator: Any,
    use_postshuffle_norm: bool,
) -> None:
    """Test VisionPatchMerger with both use_postshuffle_norm options."""

    torch.manual_seed(42)

    # Generate test input
    batch_size = 2
    seq_len = 16  # Should be divisible by spatial_merge_size^2
    spatial_merge_size = vision_config["spatial_merge_size"]
    hidden_size = vision_config["hidden_size"]

    # Ensure seq_len is divisible by spatial_merge_size^2
    while seq_len % (spatial_merge_size**2) != 0:
        seq_len += 1

    # Create input tensor
    total_patches = batch_size * seq_len
    input_tensor = torch.randn(
        total_patches, hidden_size, dtype=torch.bfloat16
    ).cuda()

    # Generate weights
    weights = weight_generator.generate_vision_patch_merger_weights(
        use_postshuffle_norm=use_postshuffle_norm
    )

    # Generate reference output using HuggingFace
    torch_output = generate_torch_outputs(
        input_tensor=input_tensor,
        weights=weights,
        vision_config=vision_config,
        use_postshuffle_norm=use_postshuffle_norm,
        dtype=torch.bfloat16,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        input_tensor=input_tensor,
        weights=weights,
        vision_config=vision_config,
        use_postshuffle_norm=use_postshuffle_norm,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Verify output shape
    expected_shape = (
        total_patches // (spatial_merge_size**2),
        vision_config["out_hidden_size"],
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )
    assert torch_output.shape == max_output.shape, (
        f"Shape mismatch: torch shape {torch_output.shape} vs max shape {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output=torch_output,
        max_output=max_output,
        rtol=RTOL,
        atol=ATOL,
        message=f"VisionPatchMerger output mismatch (use_postshuffle_norm={use_postshuffle_norm})",
    )
