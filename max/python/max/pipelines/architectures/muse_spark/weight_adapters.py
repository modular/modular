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

"""Weight adapters for Muse Spark ModuleV3."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

MUSE_SPARK_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "language_model.",
    "lm_head.weight": "language_model.lm_head.weight",
}

MUSE_SPARK_VISION_SAFETENSOR_MAP: dict[str, str] = {
    # A bare Tensor in the module, an nn.Embedding in the checkpoint.
    "model.vision_tower.patch_embedder.position_embedding_table.weight": (
        "vision_tower.patch_embedder.position_embedding_table"
    ),
    "model.vision_tower.": "vision_tower.",
    "model.vision_adapter.": "vision_adapter.",
    "model.vision_projection.": "vision_projection.",
}

# Audio input is not served.
_DROPPED_PREFIXES = ("model.audio_tower.",)

# `MoEGate` holds the router as `gate_score`, and `experts` is a ModuleList, so
# the norm shared by every expert moves up to the MoE.
_MOE_RENAMES = (
    (".mlp.gate.weight", ".mlp.gate.gate_score.weight"),
    (".mlp.experts.post_expert_norm.", ".mlp.post_expert_norm."),
)

_STACKED_EXPERTS = re.compile(
    r"^(?P<prefix>language_model\.layers\.\d+\.mlp\.experts\.)"
    r"(?P<proj>gate_up_proj|down_proj)(?P<suffix>.*)$"
)
# An NVFP4 stack's checkpoint suffixes and their leaves in ``StackedExperts``;
# a BF16 stack is a bare tensor under the checkpoint's own name.
_LEAVES = {
    "": "weight",
    "_weight_scale": "weight_scale",
    "_weight_scale_2": "weight_scale_2",
    "_input_scale": "input_scale",
}


def _rename(
    state_dict: Mapping[str, Weights],
    name_map: dict[str, str],
    skip_prefixes: tuple[str, ...],
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        if weight_name.startswith(skip_prefixes + _DROPPED_PREFIXES):
            continue
        for before, after in name_map.items():
            if weight_name.startswith(before):
                max_name = after + weight_name.removeprefix(before)
                new_state_dict[max_name] = value.data()
                break
        else:
            raise ValueError(f"Unexpected checkpoint key: {weight_name}")
    return new_state_dict


def _convert_moe(state_dict: dict[str, WeightData]) -> dict[str, WeightData]:
    out: dict[str, WeightData] = {}
    stacks: dict[str, dict[str, WeightData]] = defaultdict(dict)
    for name, data in state_dict.items():
        for before, after in _MOE_RENAMES:
            name = name.replace(before, after)
        match = _STACKED_EXPERTS.match(name)
        if match is None:
            out[name] = data
            continue
        if match["suffix"] not in _LEAVES:
            raise ValueError(f"Unexpected expert tensor: {name}")
        stacks[match["prefix"] + match["proj"]][match["suffix"]] = data

    for stack, leaves in stacks.items():
        if set(leaves) == {""}:
            out[stack] = leaves[""]
        elif set(leaves) == set(_LEAVES):
            for suffix, data in leaves.items():
                out[f"{stack}.{_LEAVES[suffix]}"] = data
        else:
            missing = sorted(set(_LEAVES) - set(leaves))
            raise ValueError(
                f"{stack} is neither BF16 nor NVFP4: missing"
                f" {[stack + s for s in missing]}"
            )
    return out


def convert_safetensor_language_state_dict(
    state_dict: Mapping[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Maps the text checkpoint keys onto the ``MuseSpark`` module tree.

    Each layer's stacked ``experts.gate_up_proj`` and ``experts.down_proj``
    stay whole: a BF16 tensor under its own name, or NVFP4 ``weight``,
    ``weight_scale``, ``weight_scale_2`` and ``input_scale`` on the layers
    the checkpoint quantizes, as ``StackedExperts`` names them.

    Raises:
        ValueError: If a key is neither a text, vision nor audio key, or an
            expert stack is not one of the two forms.
            ``Module.compile(weights=...)`` ignores extra keys, so this is
            where an unmapped checkpoint key surfaces.
    """
    return _convert_moe(
        _rename(
            state_dict,
            MUSE_SPARK_LANGUAGE_SAFETENSOR_MAP,
            tuple(MUSE_SPARK_VISION_SAFETENSOR_MAP),
        )
    )


def convert_safetensor_vision_state_dict(
    state_dict: Mapping[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Maps the vision checkpoint keys onto the vision tower's module tree.

    Raises:
        ValueError: If a key is neither a text, vision nor audio key.
    """
    return _rename(
        state_dict,
        MUSE_SPARK_VISION_SAFETENSOR_MAP,
        tuple(MUSE_SPARK_LANGUAGE_SAFETENSOR_MAP),
    )


def convert_registered_state_dict(
    state_dict: Mapping[str, Weights], **kwargs: object
) -> dict[str, WeightData]:
    """Adapts each checkpoint tensor once into its registered role namespace."""
    language = convert_safetensor_language_state_dict(state_dict, **kwargs)
    vision = convert_safetensor_vision_state_dict(state_dict, **kwargs)
    return {
        **{"decoder." + name: value for name, value in language.items()},
        **{"vision." + name: value for name, value in vision.items()},
    }
