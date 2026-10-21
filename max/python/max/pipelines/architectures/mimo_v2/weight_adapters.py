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
"""Names Xiaomi's MiMo-V2.6-Flash checkpoint tensors as MAX's weights.

Every tensor loads as the checkpoint stores it; the model reads Xiaomi's
layouts itself. The vision, audio and MTP tensors are not loaded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from max.dtype import DType
from max.graph.weights import WeightData, Weights

_IGNORED_PREFIXES = (
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
    "model.mtp.",
)


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Renames the checkpoint's text-model tensors to MAX's MiMo-V2 weights.

    Args:
        state_dict: Every tensor in the checkpoint shards, by checkpoint name.

    Returns:
        The weights keyed by MAX name: the checkpoint name without
        ``model.``, with each FP8 projection's ``weight_scale_inv`` as
        ``weight_scale`` and the router's ``mlp.gate.weight`` as
        ``mlp.gate.gate_score.weight``. The experts' ``weight_scale`` bytes
        are E8M0, which the checkpoint stores as ``U8``.
    """
    adapted: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        if name.startswith(_IGNORED_PREFIXES):
            continue
        data = weight.data()
        max_name = name.removeprefix("model.")
        if max_name.endswith(".weight_scale_inv"):
            max_name = max_name.removesuffix("_inv")
        elif max_name.endswith(".mlp.gate.weight"):
            max_name = max_name.replace(".gate.", ".gate.gate_score.")
        elif ".mlp.experts." in max_name and max_name.endswith(".weight_scale"):
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)
        adapted[max_name] = dataclasses.replace(data, name=max_name)
    return adapted
