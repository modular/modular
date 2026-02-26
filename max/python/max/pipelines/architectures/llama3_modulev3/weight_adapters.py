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

from __future__ import annotations

from max.graph.weights import WeightData, Weights

# Maps from Safetensor to MAX weight names.
# The new ModuleV3 model uses "language_model." prefix (matching GPT-OSS pattern).
LLAMA_SAFETENSOR_MAPPING = {
    "model.": "language_model.",
    "lm_head": "language_model.lm_head",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Remaps HuggingFace weight names to the MAX naming convention
    (adds ``language_model.`` prefix).
    """
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in LLAMA_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict


# Maps from GGUF to MAX weight names.
LLAMA_GGUF_MAPPING = {
    "token_embd": "language_model.embed_tokens",
    "blk": "language_model.layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    "output.weight": "language_model.lm_head.weight",
    "output_norm": "language_model.norm",
}


def convert_gguf_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert GGUF state dict to MAX format."""
    new_state_dict: dict[str, WeightData] = {}
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in LLAMA_GGUF_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    # GGUF bakes `rope_freqs` into the weights file; we compute them at runtime.
    new_state_dict.pop("rope_freqs.weight", None)

    return new_state_dict
