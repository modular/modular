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

"""Weight adapters for Muse Glimmer ModuleV3."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights

MUSE_GLIMMER_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "language_model.",
    "lm_head.weight": "language_model.lm_head.weight",
}

# TODO: map these once the vision encoder exists.
MUSE_GLIMMER_VISION_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
)


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Renames the text checkpoint keys onto the ``MuseGlimmer`` module tree.

    The checkpoint is already bf16, ``[out, in]`` and rotate-half, so the
    tensors pass through unchanged.

    Raises:
        ValueError: If a key is neither a text nor a vision key.
            ``Module.compile(weights=...)`` ignores extra keys, so this is
            where an unmapped checkpoint key surfaces.
    """
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        if weight_name.startswith(MUSE_GLIMMER_VISION_PREFIXES):
            continue
        for before, after in MUSE_GLIMMER_LANGUAGE_SAFETENSOR_MAP.items():
            if weight_name.startswith(before):
                max_name = after + weight_name.removeprefix(before)
                new_state_dict[max_name] = value.data()
                break
        else:
            raise ValueError(f"Unexpected checkpoint key: {weight_name}")
    return new_state_dict
