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

import re

from max.graph.weights import WeightData, Weights

GEMMA4_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "",
    "language_model.model.": "",
}

# For the vision model
GEMMA4_MULTIMODAL_SAFETENSOR_MAP: dict[str, str] = {
    "model.vision_tower.": "",
    "model.embed_vision": "embed_vision",
    ".linear.": ".",
}

# Regex to extract the layer index from attention projection weight names.
_ATTN_PROJ_RE = re.compile(
    r"(?P<prefix>layers\.(?P<idx>\d+)\.self_attn\.)"
    r"(?P<proj>[qkv])_proj\."
    r"(?P<suffix>.*)"
)


def _remap_attn_proj(
    name: str,
    layer_types: list[str] | None,
    attention_k_eq_v: bool,
) -> str:
    """Remap ``self_attn.{q,k,v}_proj.`` to ``self_attn.{qkv,qk}_proj.{q,k,v}.``.

    For layers where ``attention_k_eq_v`` is ``True`` and the layer is **not**
    a sliding-attention layer, ``q_proj``/``k_proj`` map to ``qk_proj.{q,k}``.
    Otherwise they map to ``qkv_proj.{q,k,v}``.
    """
    m = _ATTN_PROJ_RE.search(name)
    if m is None:
        return name

    layer_idx = int(m.group("idx"))
    proj = m.group("proj")

    # Determine whether this layer uses qk_proj (no v) or qkv_proj.
    is_sliding = True  # default: assume sliding (uses qkv_proj)
    if layer_types is not None and layer_idx < len(layer_types):
        is_sliding = layer_types[layer_idx] == "sliding_attention"

    use_qk_only = attention_k_eq_v and not is_sliding

    stacked_name = "qk_proj" if use_qk_only else "qkv_proj"
    new_proj = f"{stacked_name}.{proj}."
    return name[: m.start("proj")] + new_proj + m.group("suffix")


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
    layer_types: list[str] | None = None,
    attention_k_eq_v: bool = True,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith("language_model.") or weight_name.startswith(
            "model.language_model."
        ):
            max_name = weight_name
            for before, after in GEMMA4_LANGUAGE_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            max_name = _remap_attn_proj(max_name, layer_types, attention_k_eq_v)
            new_state_dict[max_name] = value.data()

    return new_state_dict


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model."""
    new_state_dict: dict[str, WeightData] = {}

    # include vision tower weights AND multi modal weights
    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith("model.vision_tower.")
            or weight_name.startswith("model.embed_vision.")
        ):
            continue

        max_name = weight_name

        for before, after in GEMMA4_MULTIMODAL_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        new_state_dict[max_name] = weight_data

    return new_state_dict
