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
from collections.abc import Iterator, Mapping

from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape

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
_NVFP4_SUFFIXES = frozenset(
    {"", "_weight_scale", "_weight_scale_2", "_input_scale"}
)


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


def _view(
    data: WeightData, index: tuple[int | slice, ...], name: str
) -> WeightData:
    buf = data.to_buffer()
    # Buffer indexing takes one index per axis; a 1-D scalar stack drops `i`.
    buf = buf[(index + (slice(None),) * buf.rank)[: buf.rank]]
    return WeightData(buf, name, data.dtype, Shape(buf.shape))


def _split_experts(
    prefix: str, proj: str, leaves: dict[str, WeightData]
) -> Iterator[tuple[str, WeightData]]:
    """Yields one expert's projection per row ``e`` of the stacked tensors.

    ``gate_up_proj`` stacks gate rows then up rows; its ``[E, 2]`` NVFP4
    scalars are ``(gate, up)``. Every leaf is a view, never a copy.
    """
    halves = ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
    stacked = leaves[""]
    rows = int(stacked.shape[1]) // len(halves)
    for e in range(int(stacked.shape[0])):
        for i, half in enumerate(halves):
            row_slice = slice(i * rows, (i + 1) * rows)
            for leaf, index in (
                ("weight", (e, row_slice)),
                ("weight_scale", (e, row_slice)),
                ("weight_scale_2", (e, i)),
                ("input_scale", (e, i)),
            ):
                data = leaves.get("" if leaf == "weight" else f"_{leaf}")
                if data is not None:
                    name = f"{prefix}{e}.{half}.{leaf}"
                    yield name, _view(data, index, name)


def _convert_moe(state_dict: dict[str, WeightData]) -> dict[str, WeightData]:
    out: dict[str, WeightData] = {}
    stacks: dict[tuple[str, str], dict[str, WeightData]] = defaultdict(dict)
    for name, data in state_dict.items():
        for before, after in _MOE_RENAMES:
            name = name.replace(before, after)
        match = _STACKED_EXPERTS.match(name)
        if match is None:
            out[name] = data
            continue
        if match["suffix"] not in _NVFP4_SUFFIXES:
            raise ValueError(f"Unexpected expert tensor: {name}")
        stacks[match["prefix"], match["proj"]][match["suffix"]] = data

    for (prefix, proj), leaves in stacks.items():
        if set(leaves) not in ({""}, _NVFP4_SUFFIXES):
            missing = sorted(_NVFP4_SUFFIXES - set(leaves))
            raise ValueError(
                f"{prefix}{proj} is neither BF16 nor NVFP4: missing"
                f" {[proj + s for s in missing]}"
            )
        out.update(_split_experts(prefix, proj, leaves))
    return out


def convert_safetensor_language_state_dict(
    state_dict: Mapping[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Maps the text checkpoint keys onto the ``MuseSpark`` module tree.

    Each layer's stacked ``experts.gate_up_proj`` and ``experts.down_proj``
    become per-expert ``gate_proj``/``up_proj``/``down_proj`` weights:
    BF16 ``weight`` tensors, or NVFP4 ``weight``, ``weight_scale``,
    ``weight_scale_2`` and ``input_scale`` on the layers the checkpoint
    quantizes, the names the checkpoint gives a quantized projection.

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
