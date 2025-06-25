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
from max.pipelines.architectures.internvl.internvl import (
    InternVisionEncoderLayer,
)
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
    VisionConfig,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.internvl.modeling_internvl import (
    InternVLVisionLayer,
)


@torch.no_grad()
def generate_torch_outputs(
    vision_config: VisionConfig,
    input_tensor: torch.Tensor,
    encoder_layer_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL implementation."""

    # Create a minimal config object that matches what InternVLVisionLayer expects
    class MinimalVisionConfig:
        def __init__(self, vision_config: VisionConfig) -> None:
            self.hidden_size = vision_config.hidden_size
            self.num_attention_heads = vision_config.num_attention_heads
            self.intermediate_size = vision_config.intermediate_size
            self.use_qk_norm = vision_config.qk_normalization
            self.layer_norm_eps = vision_config.layer_norm_eps
            self._attn_implementation = "eager"  # Use eager attention
            self.attention_dropout = 0.0
            self.projection_dropout = 0.0
            self.attention_bias = False
            self.hidden_dropout_prob = (
                0.0  # Our implementation doesn't have dropout
            )
            self.layer_scale_init_value = 1.0  # Default layer scale value
            self.chunk_size_feed_forward = 0  # Not used in our case
            self.norm_type = vision_config.norm_type
            self.hidden_act = "gelu"  # Default activation for MLP

    config = MinimalVisionConfig(vision_config)

    # Create the HuggingFace encoder layer
    layer = InternVLVisionLayer(config).to(torch.bfloat16).to("cuda")

    # Load attention weights - handle stacked QKV weight splitting
    qkv_weight = (
        encoder_layer_weights["attn.qkv_proj.weight"]
        .to(torch.bfloat16)
        .to("cuda")
    )
    hidden_size = vision_config.hidden_size

    # Split the stacked QKV weight
    q_weight = qkv_weight[:hidden_size, :]
    k_weight = qkv_weight[hidden_size : 2 * hidden_size, :]
    v_weight = qkv_weight[2 * hidden_size :, :]

    # Set attention weights
    layer.attention.q_proj.weight.data = q_weight
    layer.attention.k_proj.weight.data = k_weight
    layer.attention.v_proj.weight.data = v_weight
    layer.attention.projection_layer.weight.data = (
        encoder_layer_weights["attn.o_proj.weight"]
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Set normalization weights for attention
    layer.attention.q_norm.weight.data = (
        encoder_layer_weights["attn.q_norm.weight"]
        .to(torch.bfloat16)
        .to("cuda")
    )
    layer.attention.k_norm.weight.data = (
        encoder_layer_weights["attn.k_norm.weight"]
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Set layer normalization weights
    layer.layernorm_before.weight.data = (
        encoder_layer_weights["norm1.weight"].to(torch.bfloat16).to("cuda")
    )
    layer.layernorm_after.weight.data = (
        encoder_layer_weights["norm2.weight"].to(torch.bfloat16).to("cuda")
    )

    # Set MLP weights (HuggingFace InternVLVisionMLP uses fc1/fc2 naming)
    if hasattr(layer.mlp, "fc1") and hasattr(layer.mlp, "fc2"):
        layer.mlp.fc1.weight.data = (
            encoder_layer_weights["mlp.gate_proj.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.mlp.fc2.weight.data = (
            encoder_layer_weights["mlp.down_proj.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        # Set bias if exists
        if hasattr(layer.mlp.fc1, "bias") and layer.mlp.fc1.bias is not None:
            layer.mlp.fc1.bias.data = (
                encoder_layer_weights["mlp.gate_proj.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
        if hasattr(layer.mlp.fc2, "bias") and layer.mlp.fc2.bias is not None:
            layer.mlp.fc2.bias.data = (
                encoder_layer_weights["mlp.down_proj.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
    elif (
        hasattr(layer.mlp, "gate_proj")
        and hasattr(layer.mlp, "down_proj")
        and hasattr(layer.mlp, "up_proj")
    ):
        layer.mlp.gate_proj.weight.data = (
            encoder_layer_weights["mlp.gate_proj.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.mlp.down_proj.weight.data = (
            encoder_layer_weights["mlp.down_proj.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.mlp.up_proj.weight.data = (
            encoder_layer_weights["mlp.up_proj.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        # Set bias if exists
        if (
            hasattr(layer.mlp.gate_proj, "bias")
            and layer.mlp.gate_proj.bias is not None
        ):
            layer.mlp.gate_proj.bias.data = (
                encoder_layer_weights["mlp.gate_proj.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
        if (
            hasattr(layer.mlp.down_proj, "bias")
            and layer.mlp.down_proj.bias is not None
        ):
            layer.mlp.down_proj.bias.data = (
                encoder_layer_weights["mlp.down_proj.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
        if (
            hasattr(layer.mlp.up_proj, "bias")
            and layer.mlp.up_proj.bias is not None
        ):
            layer.mlp.up_proj.bias.data = (
                encoder_layer_weights["mlp.up_proj.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
    else:
        # Fallback: try to find the linear layers
        linear_layers = [
            m
            for m in layer.mlp.modules()
            if isinstance(m, torch.nn.Linear) and m != layer.mlp
        ]
        if len(linear_layers) >= 3:
            linear_layers[0].weight.data = (
                encoder_layer_weights["mlp.gate_proj.weight"]
                .to(torch.bfloat16)
                .to("cuda")
            )
            linear_layers[1].weight.data = (
                encoder_layer_weights["mlp.down_proj.weight"]
                .to(torch.bfloat16)
                .to("cuda")
            )
            linear_layers[2].weight.data = (
                encoder_layer_weights["mlp.up_proj.weight"]
                .to(torch.bfloat16)
                .to("cuda")
            )

    # Set layer scale parameters
    layer.lambda_1.data = (
        encoder_layer_weights["ls1"].to(torch.bfloat16).to("cuda")
    )
    layer.lambda_2.data = (
        encoder_layer_weights["ls2"].to(torch.bfloat16).to("cuda")
    )

    # Zero out biases if they exist (to match MAX implementation)
    for module in layer.modules():
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    # Forward pass
    output, _ = layer(input_tensor, output_attentions=False)

    return output


def generate_max_outputs(
    internvl_config: InternVLConfig,
    input_tensor: torch.Tensor,
    encoder_layer_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVisionEncoderLayer implementation."""
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Prepare state dict and map SwiGLU weight names to simple MLP weight names
    state_dict = {}
    for weight_name, value in encoder_layer_weights.items():
        # Map SwiGLU MLP naming to simple MLP naming for InternVL
        if weight_name == "mlp.gate_proj.weight":
            # gate_proj in SwiGLU corresponds to fc1 in simple MLP
            state_dict["mlp.fc1.weight"] = value.cpu()
        elif weight_name == "mlp.gate_proj.bias":
            state_dict["mlp.fc1.bias"] = value.cpu()
        elif weight_name == "mlp.down_proj.weight":
            # down_proj in SwiGLU corresponds to fc2 in simple MLP
            state_dict["mlp.fc2.weight"] = value.cpu()
        elif weight_name == "mlp.down_proj.bias":
            state_dict["mlp.fc2.bias"] = value.cpu()
        elif weight_name == "mlp.up_proj.weight":
            # Skip up_proj since simple MLP doesn't use it
            continue
        elif weight_name == "mlp.up_proj.bias":
            # Skip up_proj bias since simple MLP doesn't use it
            continue
        else:
            state_dict[weight_name] = value.cpu()

    # If using RMSNorm, remove bias keys since RMSNorm does not have bias
    if internvl_config.vision_config.norm_type == "rms_norm":
        state_dict.pop("norm1.bias", None)
        state_dict.pop("norm2.bias", None)

    # Create the MAX encoder layer
    encoder_layer = InternVisionEncoderLayer(internvl_config)

    # Load weights
    encoder_layer.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator(0)])

    # Build the graph
    batch_size, seq_len, hidden_size = input_tensor.shape
    input_type = TensorType(
        dtype, [batch_size, seq_len, hidden_size], device=device_ref
    )

    with Graph("InternVisionEncoderLayer", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = encoder_layer(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=encoder_layer.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(input_tensor).to(device))
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


def test_vision_encoder_layer(
    internvl_config: InternVLConfig,
    vision_input_tensor: torch.Tensor,
    vision_encoder_layer_weights: dict[str, torch.Tensor],
) -> None:
    """Test InternVisionEncoderLayer against PyTorch reference."""
    # TODO: Remove this once we figure out the attention error on AMD GPUs.
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    # Generate reference output
    torch_output = generate_torch_outputs(
        internvl_config.vision_config,
        vision_input_tensor,
        vision_encoder_layer_weights,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        internvl_config=internvl_config,
        input_tensor=vision_input_tensor,
        encoder_layer_weights=vision_encoder_layer_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Compare outputs
    torch.testing.assert_close(
        torch_output.squeeze(0).to(torch.bfloat16),
        max_output.squeeze(0).to(torch.bfloat16),
        rtol=4 * torch.finfo(torch.bfloat16).eps,
        atol=16 * torch.finfo(torch.bfloat16).eps,
    )
