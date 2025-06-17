# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import max.driver as md
import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.layers.attention import (
    InternVLMultiheadAttention,
)
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
    VisionConfig,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.internvl.modeling_internvl import (
    InternVLVisionAttention,
)


@torch.no_grad()
def generate_torch_outputs(
    vision_config: VisionConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL implementation."""

    # Create a minimal config object that matches what InternVLVisionAttention expects
    class MinimalConfig:
        def __init__(self, vision_config: VisionConfig):
            self.hidden_size = vision_config.hidden_size
            self.num_attention_heads = vision_config.num_attention_heads
            self.use_qk_norm = vision_config.qk_normalization
            self.layer_norm_eps = vision_config.layer_norm_eps
            self._attn_implementation = "eager"  # Use eager attention
            # Add missing attributes that HuggingFace InternVLVisionAttention expects
            self.attention_dropout = 0.0  # Default value
            self.projection_dropout = 0.0  # Default value
            self.attention_bias = False  # Default value

    config = MinimalConfig(vision_config)

    # Create the HuggingFace attention layer
    layer = InternVLVisionAttention(config).to(torch.bfloat16).to("cuda")

    # Load weights - handle stacked QKV weight splitting
    qkv_weight = (
        attention_weights["qkv_proj.weight"].to(torch.bfloat16).to("cuda")
    )
    hidden_size = vision_config.hidden_size

    # Split the stacked QKV weight
    q_weight = qkv_weight[:hidden_size, :]
    k_weight = qkv_weight[hidden_size : 2 * hidden_size, :]
    v_weight = qkv_weight[2 * hidden_size :, :]

    # Set the weights
    layer.q_proj.weight.data = q_weight
    layer.k_proj.weight.data = k_weight
    layer.v_proj.weight.data = v_weight
    layer.projection_layer.weight.data = (
        attention_weights["o_proj.weight"].to(torch.bfloat16).to("cuda")
    )

    # Set normalization weights
    layer.q_norm.weight.data = (
        attention_weights["q_norm.weight"].to(torch.bfloat16).to("cuda")
    )
    layer.k_norm.weight.data = (
        attention_weights["k_norm.weight"].to(torch.bfloat16).to("cuda")
    )

    # Zero out biases if they exist (to match MAX implementation with has_bias=False)
    if hasattr(layer.q_proj, "bias") and layer.q_proj.bias is not None:
        layer.q_proj.bias.data.zero_()
    if hasattr(layer.k_proj, "bias") and layer.k_proj.bias is not None:
        layer.k_proj.bias.data.zero_()
    if hasattr(layer.v_proj, "bias") and layer.v_proj.bias is not None:
        layer.v_proj.bias.data.zero_()
    if (
        hasattr(layer.projection_layer, "bias")
        and layer.projection_layer.bias is not None
    ):
        layer.projection_layer.bias.data.zero_()

    # Forward pass - the HuggingFace implementation returns a tuple (output, attn_weights)
    # We only need the output for comparison
    outputs = layer(input_tensor, output_attentions=False)
    output = outputs[0] if isinstance(outputs, tuple) else outputs

    return output


def generate_max_outputs(
    num_attention_heads: int,
    hidden_size: int,
    qk_normalization: bool,
    layer_norm_eps: float,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVL attention implementation."""
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Prepare state dict
    state_dict = {}
    for weight_name, value in attention_weights.items():
        state_dict[weight_name] = value.cpu()

    # Create the attention layer with individual parameters
    attention = InternVLMultiheadAttention(
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        devices=[device_ref],
        dtype=dtype,
        qk_normalization=qk_normalization,
        layer_norm_eps=layer_norm_eps,
        has_bias=False,
        stacked_qkv=True,
    )
    attention.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator(0)])

    # Build the graph
    batch_size, seq_len, hidden_size = input_tensor.shape
    input_type = TensorType(
        dtype, [batch_size, seq_len, hidden_size], device=device_ref
    )

    with Graph("InternVLVisionAttention", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = attention(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(input_tensor).to(device))
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


def test_vision_attention(
    internvl_config: InternVLConfig,
    vision_input_tensor: torch.Tensor,
    vision_attention_weights: dict[str, torch.Tensor],
) -> None:
    """Test InternVL vision attention against PyTorch reference."""
    # TODO: Remove this once we figure out the attention error on AMD GPUs.
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    # Generate reference output
    torch_output = generate_torch_outputs(
        internvl_config.vision_config,
        vision_input_tensor,
        vision_attention_weights,
    )

    # Generate MAX output using individual parameters from config
    vision_config = internvl_config.vision_config
    max_output = generate_max_outputs(
        num_attention_heads=vision_config.num_attention_heads,
        hidden_size=vision_config.hidden_size,
        qk_normalization=vision_config.qk_normalization,
        layer_norm_eps=vision_config.layer_norm_eps,
        input_tensor=vision_input_tensor,
        attention_weights=vision_attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Compare outputs
    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        max_output.squeeze(0).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
