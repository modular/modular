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

"""Weight adapters for Kimi K2.5 vision components.

Maps HuggingFace checkpoint keys to the MAX module state_dict keys for:
- Language model (``language_model.*`` -> DeepseekV3)
- Vision tower (``vision_tower.*`` → PatchEmbedding, Encoder)
- MM projector (``mm_projector.*`` → PatchMergerMLP)
"""

from __future__ import annotations

import re

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# Maps from Safetensor to MAX weight names.
DEEPSEEK_SAFETENSOR_MAP = {
    "language_": "",
    "model.": "",  # Removes the "model" prefix.
    "gate.weight": "gate.gate_score.weight",
    "weight_scale_inv": "weight_scale",
}


def convert_kimik2_5_language_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    language_state_dict: dict[str, WeightData] = {}

    # Map the weight names.
    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        language_state_dict[max_name] = value.data()

    # TODO(E2EOPT-673): Support MTP. We currently delete the MTP weights
    # This is also done in the official DeepSeek HF checkpoint converter:
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/4592be48c07f036b32ef971474068aebc489e3e7/inference/convert.py#L53-L54
    mtp_layer_idx = huggingface_config.num_hidden_layers
    for key in list(language_state_dict.keys()):
        if key.startswith(f"layers.{mtp_layer_idx}."):
            del language_state_dict[key]
        if key.endswith(".self_attn.rotary_emb.inv_freq"):  # TODO:!!!
            del language_state_dict[key]
        if key.endswith(".k_scale") or key.endswith(".v_scale"):
            del language_state_dict[key]

    return language_state_dict


# Maps from HuggingFace checkpoint names (after stripping the ``vision_tower.``
# prefix) to MAX vision tower module weight names.
#
# The ``vision_tower.`` prefix is stripped first, then these replacements are
# applied in order.
KIMIK2_5_VISION_TOWER_MAPPING = {
    "encoder.final_layernorm.": "encoder.norm.",
    ".mlp.fc0.": ".mlp.up_proj.",
    ".mlp.fc1.": ".mlp.down_proj.",
}

# Regex patterns for attention renames that need the block index captured.
# HF: ``blocks.N.wqkv.*`` → MAX: ``blocks.N.attn.wqkv.*``
# HF: ``blocks.N.wo.*``   → MAX: ``blocks.N.attn.wo.*``
_ATTN_RENAME_PATTERNS = [
    (re.compile(r"blocks\.(\d+)\.wqkv\."), r"blocks.\1.attn.wqkv."),
    (re.compile(r"blocks\.(\d+)\.wo\."), r"blocks.\1.attn.wo."),
]


def convert_kimik2_5_vision_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Kimi K2.5 vision tower weights for MAX modules.

    Filters to ``vision_tower.*`` keys, strips the prefix, and renames
    checkpoint keys to match the MAX module hierarchy (PatchEmbedding,
    Encoder with Attention/MLP2).

    Args:
        state_dict: The raw Kimi K2.5 checkpoint weights.

    Returns:
        The filtered and mapped weights for the vision tower.
    """
    vision_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if not checkpoint_name.startswith("vision_tower."):
            continue

        name = checkpoint_name[len("vision_tower.") :]

        for before, after in KIMIK2_5_VISION_TOWER_MAPPING.items():
            name = name.replace(before, after)

        for pattern, replacement in _ATTN_RENAME_PATTERNS:
            name = pattern.sub(replacement, name)

        weight_data = weight.data()

        # Cast floating-point weights to bfloat16 (skip FP8 and scale tensors).
        if weight_data.dtype.is_float():
            is_scale = checkpoint_name.endswith(
                ".weight_scale"
            ) or checkpoint_name.endswith(".input_scale")

            if not weight_data.dtype.is_float8() and not is_scale:
                weight_data = weight_data.astype(DType.bfloat16)

        vision_state_dict[name] = weight_data

    return vision_state_dict


# Maps from HuggingFace checkpoint names (after stripping the
# ``mm_projector.`` prefix) to MAX PatchMergerMLP weight names.
KIMIK2_5_MM_PROJECTOR_MAPPING = {
    "proj.0.": "linear1.",
    "proj.2.": "linear2.",
}


def convert_kimik2_5_mm_projector_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Kimi K2.5 mm_projector weights for PatchMergerMLP.

    Filters to ``mm_projector.*`` keys, strips the prefix, and renames
    Sequential indices to the named attributes used by PatchMergerMLP.

    Args:
        state_dict: The raw Kimi K2.5 checkpoint weights.

    Returns:
        The filtered and mapped weights for PatchMergerMLP.
    """
    projector_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if not checkpoint_name.startswith("mm_projector."):
            continue

        name = checkpoint_name[len("mm_projector.") :]

        for before, after in KIMIK2_5_MM_PROJECTOR_MAPPING.items():
            name = name.replace(before, after)

        weight_data = weight.data()

        if weight_data.dtype.is_float():
            is_scale = checkpoint_name.endswith(
                ".weight_scale"
            ) or checkpoint_name.endswith(".input_scale")

            if not weight_data.dtype.is_float8() and not is_scale:
                weight_data = weight_data.astype(DType.bfloat16)

        projector_state_dict[name] = weight_data

    return projector_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    **kwargs,
) -> dict[str, WeightData]:
    """Convert full Kimi K2.5 safetensor state dict for the MAX pipeline.

    Calls the vision, projector, and language converters then merges their
    outputs into a single state dict whose keys match the KimiK2_5 module
    (vision_encoder.*, multimodal_projector.*, language_model.*). The
    pipeline passes this single dict to the model; load_model splits it by
    prefix for each compiled graph.

    Args:
        state_dict: Raw checkpoint as dict from weight name to Weights.
        huggingface_config: HuggingFace config (required for language MTP pruning).
        **kwargs: Forwarded to the per-component converters.

    Returns:
        Single merged state dict with module-prefixed keys.
    """
    if huggingface_config is None:
        raise ValueError(
            "convert_safetensor_state_dict requires huggingface_config for "
            "language model weight conversion (MTP layer pruning)."
        )
    # Only pass language keys so we do not mix in vision/projector keys.
    language_raw = {
        k: v for k, v in state_dict.items() if k.startswith("language_model.")
    }
    language = convert_kimik2_5_language_state_dict(
        language_raw,
        huggingface_config=huggingface_config,
        **kwargs,
    )
    vision = convert_kimik2_5_vision_state_dict(state_dict, **kwargs)
    projector = convert_kimik2_5_mm_projector_state_dict(state_dict, **kwargs)

    return (
        {f"language_model.{k}": v for k, v in language.items()}
        | {f"vision_encoder.{k}": v for k, v in vision.items()}
        | {f"multimodal_projector.{k}": v for k, v in projector.items()}
    )
