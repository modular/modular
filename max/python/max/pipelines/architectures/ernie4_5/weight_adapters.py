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

"""Weight name adapters for ERNIE-4.5 (Ernie4_5ForCausalLM)."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

# HF safetensors layout (standard):
#   model.embed_tokens.weight
#   model.layers.N.self_attn.{q,k,v,o}_proj.weight
#   model.layers.N.mlp.{gate,up,down}_proj.weight
#   model.layers.N.{input,post_attention}_layernorm.weight
#   model.norm.weight
#   lm_head.weight  (absent when tie_word_embeddings=True)

ERNIE45_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "language_model.",
}


def _target_dtype(pipeline_config: Any) -> DType | None:
    try:
        from max.pipelines.modeling.config_enums import supported_encoding_dtype

        enc = pipeline_config.model.quantization_encoding
        if enc is not None:
            return supported_encoding_dtype(enc)
    except Exception:
        pass
    return None


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config: Any = None,
    **kwargs,
) -> dict[str, WeightData]:
    """Remap HF safetensors weight names to MAX internal convention.

    The mapping applies the ERNIE-4.5 safetensor naming rules and also
    optionally casts weights to the target dtype derived from the
    ``pipeline_config`` quantization encoding.

    Args:
        state_dict: Raw HF weights keyed by safetensor name.
        huggingface_config: Optional HuggingFace config (kept for adapter
            interface compatibility).
        pipeline_config: Optional pipeline config used to determine target
            dtype for casting.

    Returns:
        Transformed weight dict keyed by MAX-internal names with values as
        ``WeightData`` objects.
    """

    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None
    has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        if has_lm_prefix:
            max_name = weight_name
        else:
            max_name = weight_name
            for before, after in ERNIE45_SAFETENSOR_MAPPING.items():
                max_name = max_name.replace(before, after)
            if max_name == "lm_head.weight":
                max_name = "language_model.lm_head.weight"

        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd
    return new_state_dict


def convert_gguf_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config: Any = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert GGUF-style weight names to MAX internal names.

    Applies the GGUF→MAX name mapping and casts weights to the pipeline
    target dtype when applicable. Removes any GGUF rope frequency weights
    which are handled separately by the code that constructs freqs_cis.

    Args:
        state_dict: Raw GGUF weights mapping.
        huggingface_config: Optional HuggingFace config (unused).
        pipeline_config: Optional pipeline config used to determine target
            dtype for casting.

    Returns:
        Converted weight dict keyed by MAX-internal names.
    """

    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None
    gguf_mapping = {
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
    new_state_dict: dict[str, WeightData] = {}
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in gguf_mapping.items():
            max_name = max_name.replace(before, after)
        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd
    new_state_dict.pop("rope_freqs.weight", None)
    return new_state_dict
