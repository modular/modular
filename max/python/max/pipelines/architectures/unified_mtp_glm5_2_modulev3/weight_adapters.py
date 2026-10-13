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
"""Splits a GLM-5.x checkpoint between a ModuleV3 target and its MTP draft."""

from __future__ import annotations

import copy

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

from ..deepseekV3_2_modulev3 import weight_adapters as target_adapters

_SHARED_HEAD_NORM = "draft.shared_head_norm.weight"
_TARGET_FINAL_NORM = "target.language_model.norm.weight"

# The MTP layer's own head, from its suffix under the layer to the draft's
# attribute. Everything else under the layer is the decoder layer.
_MTP_HEAD_MAP = {
    "shared_head.norm.": "shared_head_norm.",
    "enorm.": "enorm.",
    "hnorm.": "hnorm.",
    "eh_proj.": "eh_proj.",
}

# Shared with the target, so the draft holds no copy.
_MTP_SHARED = ("embed_tokens.", "shared_head.head.")


def convert_with_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **kwargs,
) -> dict[str, WeightData]:
    """Converts safetensor weights for the ModuleV3 GLM-5.x MTP model.

    The MTP layer (at index ``num_hidden_layers``) converts as one more target
    layer, so it takes the target's quantization layout and ModuleV3 names,
    and lands under ``draft.*``. Every other weight lands under ``target.*``.
    """
    num_layers = huggingface_config.num_hidden_layers
    # The target conversion drops the layer past the last one, which is the
    # MTP layer; one more layer keeps it.
    with_mtp = copy.copy(huggingface_config)
    with_mtp.num_hidden_layers = num_layers + 1
    mtp_prefix = f"language_model.layers.{num_layers}."

    converted: dict[str, WeightData] = {}
    for name, value in target_adapters.convert_safetensor_state_dict(
        state_dict, with_mtp, **kwargs
    ).items():
        if not name.startswith(mtp_prefix):
            converted[f"target.{name}"] = value
            continue
        suffix = name[len(mtp_prefix) :]
        if suffix.startswith(_MTP_SHARED):
            continue
        for before, after in _MTP_HEAD_MAP.items():
            if suffix.startswith(before):
                converted[f"draft.{after}{suffix[len(before) :]}"] = value
                break
        else:
            converted[f"draft.decoder_layer.{suffix}"] = value
    # Some checkpoints share the MTP head norm with the target's final norm
    # and do not emit it under the MTP layer.
    if _SHARED_HEAD_NORM not in converted and _TARGET_FINAL_NORM in converted:
        converted[_SHARED_HEAD_NORM] = converted[_TARGET_FINAL_NORM]
    return converted
