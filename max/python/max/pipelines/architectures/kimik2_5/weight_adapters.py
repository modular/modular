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

"""Maps Kimi K2.5 checkpoints onto the graph-API modules.

:func:`kimik2_5_weight_rules` names a checkpoint's language model under
``language_model.`` and its vision tower and projector under
``vision_encoder.``, as the ``KimiK2_5`` module nests them. The Eagle3 draft
converters below stay functions: they cast the float32 and float16 drafts
some checkpoints ship to bfloat16, which rules can't.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.pipelines.lib import (
    Bitcast,
    CheckpointRole,
    Ignore,
    WeightEntry,
    WeightRenaming,
    WeightRule,
    WeightTransform,
)
from transformers.configuration_utils import PretrainedConfig

_VISION_PREFIXES = ("vision_tower.", "mm_projector.", "multi_modal_projector.")

KIMIK2_5_VISION_RENAMES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^vision_tower\.", "vision_encoder."),
    # Kimi K2.5 numbers the projector's linears as a Sequential's;
    # Kimi-VL names them.
    WeightRenaming(
        r"^(?:mm|multi_modal)_projector\.", "vision_encoder.patch_merger."
    ),
    WeightRenaming(
        r"^(vision_encoder\.patch_merger\.)(?:proj\.0|linear_1)\.",
        r"\1linear1.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.patch_merger\.)(?:proj\.2|linear_2)\.",
        r"\1linear2.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.)final_layernorm\.", r"\1norm."
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.)(wqkv|wo)\.",
        r"\1attn.\2.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.mlp\.)fc0\.",
        r"\1up_proj.",
    ),
    WeightRenaming(
        r"^(vision_encoder\.encoder\.blocks\.\d+\.mlp\.)fc1\.",
        r"\1down_proj.",
    ),
)
"""Renames the vision tower's and the projector's weights."""


def kimik2_5_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a Kimi K2.5 checkpoint onto its modules.

    The checkpoint's NextN layer, past the language model's last one, is
    left out.

    Args:
        huggingface_config: The checkpoint's config.
    """
    text_config = huggingface_config.get_text_config()
    return (
        Ignore(
            rf"^language_model\.model\.layers\.{text_config.num_hidden_layers}\.",
            reason="the NextN layer, which only multi-token prediction loads",
        ),
        Ignore(
            r"\.rotary_emb\.inv_freq$",
            reason="rotary frequencies, which the model computes",
        ),
        Ignore(
            r"\.(k|v)_scale$",
            reason="modelopt FP8 KV-cache scales, which MAX does not read",
        ),
        WeightRenaming(r"^language_model\.model\.", "language_model."),
        WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
        WeightRenaming(r"\.weight_scale_inv$", ".weight_scale"),
        *KIMIK2_5_VISION_RENAMES,
        Bitcast(r"\.weight_scale$", src=DType.uint8, dst=DType.float8_e8m0fnu),
    )


def kimik2_5_weight_entry(adapter: WeightsAdapter | None) -> WeightEntry:
    """Returns the entry that names a Kimi K2.5 checkpoint through ``adapter``.

    Args:
        adapter: The weight transform built from the architecture's rules.

    Raises:
        ValueError: If ``adapter`` is not a :class:`WeightTransform`.
    """
    if not isinstance(adapter, WeightTransform):
        raise ValueError(
            "Kimi K2.5 loads its checkpoint through its architecture's weight"
            " rules, but the model has no WeightTransform"
        )
    return WeightEntry(CheckpointRole.TARGET, adapter)


def _cast_vision_weight(checkpoint_name: str, weight: Weights) -> WeightData:
    """Return WeightData, casting floats (non-FP8, non-scale) to bfloat16."""
    weight_data = weight.data()
    is_scale = checkpoint_name.endswith((".weight_scale", ".input_scale"))
    if (
        weight_data.dtype.is_float()
        and not weight_data.dtype.is_float8()
        and not is_scale
    ):
        weight_data = weight_data.astype(DType.bfloat16)
    return weight_data


# ---------------------------------------------------------------------------
# Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Eagle3 checkpoint key prefix -> Eagle3DeepseekV3 module path.
_EAGLE3_KEY_MAP: dict[str, str] = {
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "decoder_layer.input_layernorm.",
    "layers.0.self_attn.": "decoder_layer.self_attn.",
    "layers.0.post_attention_layernorm.": "decoder_layer.post_attention_layernorm.",
    "layers.0.mlp.": "decoder_layer.mlp.",
}


def convert_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert an Eagle3 draft checkpoint to Eagle3DeepseekV3 module keys.

    The Eagle3 checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``) has keys:
    ``fc.*``, ``layers.0.*``, ``norm.*``, ``lm_head.*``.

    All weights are loaded; ``norm`` and ``lm_head`` are kept independent
    from the target model.  Only ``embed_tokens``
    is shared from the target. Float weights are cast to bfloat16, the
    head's dtype: some drafts ship float32 (``nvidia/Kimi-K2.6-Eagle3``
    before its 2026-06-11 revision).

    Args:
        **unused_kwargs: Accepted so every adapter shares one signature.
        state_dict: Raw Eagle3 checkpoint.

    Returns:
        State dict with keys matching ``Eagle3DeepseekV3`` module hierarchy.
    """
    result: dict[str, WeightData] = {}

    for name, weight in state_dict.items():
        # Apply key mapping
        mapped = False
        for before, after in _EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = _cast_vision_weight(name, weight)
                mapped = True
                break

        if not mapped:
            # fc.*, norm.*, lm_head.* pass through directly
            result[name] = _cast_vision_weight(name, weight)

    return result


# ---------------------------------------------------------------------------
# Llama-style Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Llama Eagle3 checkpoint key prefix -> Eagle3MHADraft module path.
# The MHA draft module's layer is flat (single block, no ``decoder_layer``
# namespace), so the mapping strips the single-layer prefix and inlines
# norms.
#
# Checkpoints in the wild use different single-layer prefix conventions
# (``midlayer.``, ``model.layers.0.``, ``layers.0.``). Handle all three.
_LLAMA_EAGLE3_KEY_MAP: dict[str, str] = {
    "midlayer.hidden_norm.": "hidden_norm.",
    "midlayer.input_layernorm.": "input_layernorm.",
    "midlayer.self_attn.": "self_attn.",
    "midlayer.post_attention_layernorm.": "post_attention_layernorm.",
    "midlayer.mlp.": "mlp.",
    "model.layers.0.hidden_norm.": "hidden_norm.",
    "model.layers.0.input_layernorm.": "input_layernorm.",
    "model.layers.0.self_attn.": "self_attn.",
    "model.layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "model.layers.0.mlp.": "mlp.",
    "model.norm.": "norm.",
    "model.embed_tokens.": "embed_tokens.",
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "input_layernorm.",
    "layers.0.self_attn.": "self_attn.",
    "layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "layers.0.mlp.": "mlp.",
}


def convert_llama_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``LlamaForCausalLMEagle3`` checkpoint to ``Eagle3MHADraft``.

    Handles both ``model.*``-prefixed (standard HF Llama) and
    ``layers.0.*``-prefixed (EAGLE-export) checkpoints. ``fc.*``,
    ``norm.*``, and ``lm_head.*`` pass through.
    """
    result: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        mapped = False
        for before, after in _LLAMA_EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = _cast_vision_weight(name, weight)
                mapped = True
                break
        if not mapped:
            # fc.*, norm.*, lm_head.* and any other top-level keys.
            result[name] = _cast_vision_weight(name, weight)
    return result
