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

from max.driver import Tensor
from max.graph.weights import WeightData, Weights

# Maps from Qwen2.5VL checkpoint names to Qwen2.5VLLanguageModel weight names.
QWEN2_5_VL_MODEL_MAPPING = {
    "model.": "language_model.",
    "visual.": "vision_encoder.",
    "merger.ln_q.": "merger.norm.",
    "merger.mlp.0.": "merger.linear1.",
    "merger.mlp.2.": "merger.linear2.",
}


def convert_qwen2_5vl_model_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert Qwen2.5VL model weights.

    Qwen2.5VL checkpoints have language model weights prefixed with
    `language_model.`, but Qwen2.5VLLanguageModel expects the standard Llama3
    naming without this prefix.

    This adapter:
    1. Filters to only include language model weights (those with
       `language_model.` prefix).
    2. Strips the `language_model.model.` prefix to match Qwen2.5VLLanguageModel
       expectations.
    3. Excludes vision model and multimodal projection weights.

    Args:
        state_dict: The raw Qwen2.5VL checkpoint weights.
        huggingface_config: The Qwen2.5VL HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for Qwen2.5VLLanguageModel.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Special case for lm_head. Because config.tie_word_embeddings is true
        # for some Qwen2.5VL models and false for others.
        if checkpoint_name.startswith("lm_head."):
            llm_name = checkpoint_name.replace(
                "lm_head.", "language_model.lm_head."
            )
            llm_state_dict[llm_name] = weight.data()
        elif "patch_embed.proj." in checkpoint_name:
            # Convert Conv3D weight to a Linear-equivalent format. MAX uses Linear layer instead of Conv3D for patch embedding.
            weight_data = weight.data()
            weight_array = Tensor.from_dlpack(weight_data.data)
            out_channels, in_channels, kernel_h, kernel_w, kernel_d = (
                weight_array.shape
            )
            weight_array = weight_array.view(
                dtype=weight_array.dtype,
                shape=(
                    out_channels,
                    in_channels * kernel_h * kernel_w * kernel_d,
                ),
            )
            weight_data = WeightData(
                data=weight_array,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=weight_data.shape.__class__(weight_array.shape),
                quantization_encoding=weight_data.quantization_encoding,
            )
            llm_name = "vision_encoder.patch_embed.proj.weight"
            llm_state_dict[llm_name] = weight_data
        else:
            llm_name = checkpoint_name
            for before, after in QWEN2_5_VL_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)

            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict
