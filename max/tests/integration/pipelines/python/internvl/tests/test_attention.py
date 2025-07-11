# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for InternVL attention layer."""

import os

import pytest
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVLConfig,
)
from internvl_impl.modeling_intern_vit import (
    InternAttention as InternVisionAttention,
)
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.layers.attention import (
    InternVLMultiheadAttention,
)
from max.pipelines.architectures.internvl.model_config import (
    VisionConfig,
)
from torch.utils.dlpack import from_dlpack
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_torch_outputs(
    hf_config: HFInternVLConfig,
    input_tensor: torch.Tensor,
    vision_attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL implementation."""
    # Create the HuggingFace attention layer
    layer = InternVisionAttention(hf_config).to(torch.bfloat16).to("cuda")

    # Convert weights to HuggingFace state dict format
    hf_state_dict = {}
    for k, v in vision_attention_weights.items():
        hf_state_dict[k] = v.to(torch.bfloat16).to("cuda")

    # Load weights using state_dict, handling QKV weight splitting
    layer.load_state_dict(hf_state_dict, strict=True)
    layer.eval()

    # Forward pass - the HuggingFace implementation returns a tuple (output, attn_weights)
    # We only need the output for comparison
    output = layer(input_tensor)

    return output


def generate_max_outputs(
    max_config: VisionConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVL attention implementation."""
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(attention_weights)

    # Create the attention layer with individual parameters
    head_dim = max_config.hidden_size // max_config.num_attention_heads
    attention = InternVLMultiheadAttention(
        num_attention_heads=max_config.num_attention_heads,
        hidden_size=max_config.hidden_size,
        head_dim=head_dim,
        devices=[device_ref],
        dtype=max_config.dtype,
        qk_normalization=max_config.qk_normalization,
        layer_norm_eps=max_config.layer_norm_eps,
        qkv_has_bias=max_config.qkv_bias,
        o_proj_has_bias=max_config.o_proj_bias,
        stacked_qkv=True,
    )
    attention.load_state_dict(max_weights, strict=True)

    session = InferenceSession(devices=[Accelerator(0)])

    # Build the graph
    batch_size, seq_len, hidden_size = input_tensor.shape
    input_type = TensorType(
        dtype, [batch_size, seq_len, hidden_size], device=device_ref
    )

    with Graph("InternVisionAttention", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = attention(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(input_tensor).to(device))
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
def test_vision_attention(config_name: ConfigNames) -> None:
    """Test InternVL vision attention against PyTorch reference."""
    # Create test instance and load config
    config_loader = get_config_loader()
    weight_generator = get_weight_generator(config_name)

    # Create config-specific fixtures
    internvl_config = config_loader.create_internvl_config(config_name)
    vision_config = internvl_config.vision_config

    hf_config = config_loader.load_hf_vision_config(config_name)

    # Create vision input tensor
    torch.manual_seed(42)
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    batch_size = 1
    vision_input_tensor = torch.randn(
        batch_size,
        num_patches,  # sequence length is number of patches
        vision_config.hidden_size,
        dtype=torch.bfloat16,
    ).to("cuda")

    # Generate vision attention weights
    vision_attention_weights = (
        weight_generator.generate_vision_attention_weights()
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        hf_config,
        vision_input_tensor,
        vision_attention_weights,
    )

    # Generate MAX output using individual parameters from config
    max_output = generate_max_outputs(
        max_config=vision_config,
        input_tensor=vision_input_tensor,
        attention_weights=vision_attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Compare outputs using the base class method
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision attention outputs do not match",
    )
