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

from __future__ import annotations

from max.graph.weights import WeightData, Weights

# Maps from InternVL checkpoint names to DistributedLlama3 weight names.
INTERNVL_LANGUAGE_MODEL_MAPPING = {
    # Strip the "language_model.model." prefix from most weights
    # InternVL checkpoint: "language_model.model.layers.0.input_layernorm.weight"
    # DistributedLlama3 expects: "layers.0.input_layernorm.weight"
    "language_model.model.": "",
    # Handle the output layer separately
    # InternVL checkpoint: "language_model.lm_head.weight"
    # DistributedLlama3 expects: "lm_head.weight"
    "language_model.lm_head.": "lm_head.",
}

# Maps from InternVL checkpoint names to InternVLVisionModel weight names.
INTERNVL_VISION_MODEL_MAPPING = {
    # Strip the vision_model. prefix, since InternVisionEmbeddings expects
    # "embeddings.class_embedding", for example.
    "vision_model.": "",
    # Map encoder.layers to encoder_layers to match LayerList attribute naming
    "encoder.layers.": "encoder_layers.",
    # Map attention weight names: checkpoint has "qkv" but model expects "qkv_proj"
    ".attn.qkv.": ".attn.qkv_proj.",
    ".attn.qkv_bias.": ".attn.qkv_proj_bias.",
    ".attn.proj.": ".attn.o_proj.",
    # Map mlp1 numbered layers to descriptive names
    "mlp1.0.": "mlp1.layer_norm.",  # Layer normalization
    "mlp1.1.": "mlp1.fc1.",  # First linear layer
    "mlp1.3.": "mlp1.fc2.",  # Second linear layer (note: it's 3, not 2)
}


def convert_internvl_language_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert InternVL language model weights for DistributedLlama3.

    InternVL checkpoints have language model weights prefixed with
    `language_model.`, but DistributedLlama3 expects the standard Llama3
    naming without this prefix.

    This adapter:
    1. Filters to only include language model weights (those with
       `language_model.` prefix).
    2. Strips the `language_model.model.` prefix to match DistributedLlama3
       expectations.
    3. Excludes vision model and multimodal projection weights.

    Args:
        state_dict: The raw InternVL checkpoint weights.
        huggingface_config: The InternVL HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for DistributedLlama3.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Only process language model weights (skip vision model and mlp1)
        if checkpoint_name.startswith("language_model."):
            # Apply mapping to strip "language_model." prefixes
            llm_name = checkpoint_name
            for before, after in INTERNVL_LANGUAGE_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)

            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict


def convert_internvl_vision_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert InternVL vision model weights for InternVLVisionModel.

    InternVL checkpoints have vision model weights prefixed with
    `vision_model.`, but InternVLVisionModel expects that prefix dropped.

    This adapter:
    1. Filters to only include vision model weights (those with
       `vision_model.` prefix).
    2. Strips the `vision_model.` prefix to match InternVLVisionModel
       expectations.
    3. Converts Conv2D patch embedding weights to Linear format.
    4. Excludes language model weights.

    Args:
        state_dict: The raw InternVL checkpoint weights.
        huggingface_config: The InternVL HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for InternVLVisionModel.
    """
    vision_model_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Only process vision model weights (skip language model).
        if checkpoint_name.startswith("language_model."):
            continue

        # Apply mapping to strip "vision_model." prefixes.
        vision_model_name = checkpoint_name
        for before, after in INTERNVL_VISION_MODEL_MAPPING.items():
            vision_model_name = vision_model_name.replace(before, after)

        weight_data = weight.data()

        # Convert Conv2D patch embedding weights to Linear format
        if vision_model_name == "embeddings.patch_embedding.weight":
            # Conv2D weight shape: (out_channels, in_channels, kernel_h, kernel_w)
            # For patch embedding: (embed_dim, 3, patch_size, patch_size)
            # Need to reshape to: (embed_dim, 3 * patch_size * patch_size)

            # Get the weight array
            weight_array = weight_data.data

            # Get dimensions
            out_channels, in_channels, kernel_h, kernel_w = weight_array.shape

            # Reshape from (out_channels, in_channels, kernel_h, kernel_w)
            # to (out_channels, in_channels * kernel_h * kernel_w)
            weight_array = weight_array.reshape(out_channels, -1)

            # Create new WeightData with reshaped array
            weight_data = WeightData(
                data=weight_array,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=weight_data.shape.__class__(weight_array.shape),
                quantization_encoding=weight_data.quantization_encoding,
            )

        vision_model_state_dict[vision_model_name] = weight_data

    return vision_model_state_dict
