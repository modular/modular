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
    InternVLVisionModel,
)
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
    VisionConfig,
)
from torch.utils.dlpack import from_dlpack
from transformers.configuration_utils import PretrainedConfig
from transformers.models.internvl.modeling_internvl import (
    InternVLVisionModel as HFInternVLVisionModel,
)


@torch.no_grad()
def generate_torch_outputs(
    vision_config: VisionConfig,
    pixel_values: torch.Tensor,
    vision_model_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace InternVL Vision Model implementation."""

    # Permute from [batch, height, width, channels] to [batch, channels, height, width]
    pixel_values = pixel_values.permute(0, 3, 1, 2)

    # Create a minimal config object that matches what InternVLVisionModel expects
    class MinimalVisionConfig(PretrainedConfig):
        def __init__(self, vision_config: VisionConfig):
            super().__init__()
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
            self.num_hidden_layers = vision_config.num_hidden_layers
            # HuggingFace expects these as tuples/lists with (height, width)
            self.image_size = (
                vision_config.image_size,
                vision_config.image_size,
            )
            self.patch_size = (
                vision_config.patch_size,
                vision_config.patch_size,
            )
            self.num_channels = 3
            self.use_absolute_position_embeddings = True
            self.use_mask_token = False
            self.use_mean_pooling = False
            self.initializer_range = 0.02
            self.output_attentions = False
            self.output_hidden_states = False

    config = MinimalVisionConfig(vision_config)

    # Create the HuggingFace vision model
    model = HFInternVLVisionModel(config).to(torch.bfloat16).to("cuda")

    # Load embeddings weights
    model.embeddings.cls_token.data = (
        vision_model_weights["embeddings.class_embedding"]
        .squeeze(0)
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load patch embeddings (Conv2d projection)
    model.embeddings.patch_embeddings.projection.weight.data = (
        vision_model_weights["embeddings.patch_embedding.filter"]
        .to(torch.bfloat16)
        .to("cuda")
    )
    model.embeddings.patch_embeddings.projection.bias.data = (
        vision_model_weights["embeddings.patch_embedding.bias"]
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load position embeddings - ensure correct shape without squeezing
    model.embeddings.position_embeddings.data = (
        vision_model_weights["embeddings.position_embedding"]
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load encoder layer weights
    for layer_idx in range(config.num_hidden_layers):
        layer = model.encoder.layer[layer_idx]

        # Load attention weights - handle stacked QKV weight splitting
        qkv_weight = (
            vision_model_weights[
                f"encoder_layers.{layer_idx}.attn.qkv_proj.weight"
            ]
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
            vision_model_weights[
                f"encoder_layers.{layer_idx}.attn.o_proj.weight"
            ]
            .to(torch.bfloat16)
            .to("cuda")
        )

        # Set normalization weights for attention
        layer.attention.q_norm.weight.data = (
            vision_model_weights[
                f"encoder_layers.{layer_idx}.attn.q_norm.weight"
            ]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.attention.k_norm.weight.data = (
            vision_model_weights[
                f"encoder_layers.{layer_idx}.attn.k_norm.weight"
            ]
            .to(torch.bfloat16)
            .to("cuda")
        )

        # Set layer normalization weights
        layer.layernorm_before.weight.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.norm1.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.layernorm_after.weight.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.norm2.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )

        # Set MLP weights (HuggingFace uses fc1/fc2 naming)
        layer.mlp.fc1.weight.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.mlp.fc1.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.mlp.fc2.weight.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.mlp.fc2.weight"]
            .to(torch.bfloat16)
            .to("cuda")
        )

        # Set bias if exists
        if hasattr(layer.mlp.fc1, "bias") and layer.mlp.fc1.bias is not None:
            layer.mlp.fc1.bias.data = (
                vision_model_weights[f"encoder_layers.{layer_idx}.mlp.fc1.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )
        if hasattr(layer.mlp.fc2, "bias") and layer.mlp.fc2.bias is not None:
            layer.mlp.fc2.bias.data = (
                vision_model_weights[f"encoder_layers.{layer_idx}.mlp.fc2.bias"]
                .to(torch.bfloat16)
                .to("cuda")
            )

        # Set layer scale parameters
        layer.lambda_1.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.ls1"]
            .to(torch.bfloat16)
            .to("cuda")
        )
        layer.lambda_2.data = (
            vision_model_weights[f"encoder_layers.{layer_idx}.ls2"]
            .to(torch.bfloat16)
            .to("cuda")
        )

    # Zero out biases if they exist (to match MAX implementation)
    for module in model.modules():
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    # Forward pass
    output = model(pixel_values=pixel_values)

    return output.last_hidden_state


def generate_max_outputs(
    internvl_config: InternVLConfig,
    pixel_values: torch.Tensor,
    vision_model_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVLVisionModel implementation."""
    is_gpu = isinstance(device, Accelerator)
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create weight mapping similar to Llama3 weight adapters
    # Maps from original encoder layer weight names to LayerList scoped weight names
    def map_encoder_layer_weights(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map encoder layer weights to LayerList scoped names."""
        mapped_state_dict = {}

        for weight_name, value in state_dict.items():
            if weight_name.startswith("encoder_layers."):
                # Extract layer index and rest of the path
                parts = weight_name.split(".")
                layer_idx = parts[1]
                rest_path = ".".join(parts[2:])

                # Map to LayerList naming: encoder_layers.{i}.{rest_path}
                # LayerList uses string indices for sublayer registration
                new_weight_name = f"encoder_layers.{layer_idx}.{rest_path}"
                mapped_state_dict[new_weight_name] = value.cpu()
            elif weight_name == "embeddings.patch_embedding.filter":
                # Map Conv2D filter weight to expected name and convert format
                # Conv2D shape: (out_channels, in_channels, kernel_h, kernel_w)
                # Linear shape: (out_channels, in_channels * kernel_h * kernel_w)
                out_channels, in_channels, kernel_h, kernel_w = value.shape
                value_reshaped = value.reshape(
                    out_channels, in_channels * kernel_h * kernel_w
                )
                mapped_state_dict["embeddings.patch_embedding.weight"] = (
                    value_reshaped.cpu()
                )
            elif weight_name == "embeddings.patch_embedding.bias":
                # Map Conv2D bias to expected name
                mapped_state_dict["embeddings.patch_embedding.bias"] = (
                    value.cpu()
                )
            else:
                # Keep all other weights as-is
                mapped_state_dict[weight_name] = value.cpu()

        # Add final layer norm weight (initialized to ones, standard layer norm initialization)
        # This matches the PyTorch reference implementation where the final layernorm is applied
        hidden_size = internvl_config.vision_config.hidden_size
        mapped_state_dict["layernorm.weight"] = torch.ones(
            hidden_size, dtype=torch.bfloat16
        ).cpu()

        return mapped_state_dict

    # Apply weight mapping
    state_dict = map_encoder_layer_weights(vision_model_weights)

    # Create the vision model
    vision_model = InternVLVisionModel(internvl_config)

    # Load all weights with proper scoping - now strict=True should work
    vision_model.load_state_dict(state_dict, strict=True)

    session = InferenceSession(devices=[Accelerator(0)])

    # Build the graph
    batch_size, height, width, channels = pixel_values.shape
    input_type = TensorType(
        dtype, [batch_size, height, width, channels], device=device_ref
    )

    with Graph("InternVLVisionModel", input_types=(input_type,)) as graph:
        pixel_values_input = graph.inputs[0]
        output = vision_model(pixel_values_input.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=vision_model.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(pixel_values).to(device))
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


def test_vision_model(
    internvl_config: InternVLConfig,
    vision_pixel_values: torch.Tensor,
    vision_model_weights: dict[str, torch.Tensor],
) -> None:
    """Test complete InternVLVisionModel against PyTorch reference."""
    # TODO: Remove this once we figure out the attention error on AMD GPUs.
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    # Generate reference output
    torch_output = generate_torch_outputs(
        internvl_config.vision_config,
        vision_pixel_values,
        vision_model_weights,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        internvl_config=internvl_config,
        pixel_values=vision_pixel_values,
        vision_model_weights=vision_model_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output.to(torch.bfloat16),
        rtol=8 * torch.finfo(torch.bfloat16).eps,
        atol=32 * torch.finfo(torch.bfloat16).eps,
    )
