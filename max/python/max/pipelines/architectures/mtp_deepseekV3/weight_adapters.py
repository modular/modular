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
"""Maps a DeepseekV3 checkpoint's NextN layer onto the NextN module."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

from ..deepseekV3.weight_adapters import DEEPSEEK_SAFETENSOR_MAP

__all__ = ["convert_mtp_safetensor_state_dict"]

# The NextN layer sits at index ``num_hidden_layers``, after the trunk's.
# Each suffix maps onto the NextN module's own name; the rest of the layer is
# its decoder layer.
_MTP_LAYER_MAP = {
    "shared_head.norm.": "shared_head_norm.",
    "enorm.": "enorm.",
    "hnorm.": "hnorm.",
    "eh_proj.": "eh_proj.",
}

# The layer's copies of the embedding and head, which the draft reads from
# the target instead.
_TARGET_MODULES = ("shared_head.head.", "embed_tokens.")


def convert_mtp_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Returns the NextN layer's weights, under the NextN module's names.

    The trunk's weights load through the target's own adapter, which drops
    this layer.
    """
    new_state_dict: dict[str, WeightData] = {}
    mtp_prefix = f"layers.{huggingface_config.num_hidden_layers}."
    final_norm: WeightData | None = None

    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        # Drop FP8 KV-cache static scales emitted by modelopt NVFP4
        # checkpoints, as the target's adapter does.
        if max_name.endswith((".k_scale", ".v_scale")):
            continue
        if max_name == "norm.weight":
            final_norm = value.data()
        if not max_name.startswith(mtp_prefix):
            continue
        suffix = max_name[len(mtp_prefix) :]
        if suffix.startswith(_TARGET_MODULES):
            continue
        for before, after in _MTP_LAYER_MAP.items():
            if suffix.startswith(before):
                new_state_dict[after + suffix[len(before) :]] = value.data()
                break
        else:
            new_state_dict[f"decoder_layer.{suffix}"] = value.data()

    # Some checkpoints share shared_head_norm with the base model's final
    # norm and don't emit it as a draft weight.
    if (
        "shared_head_norm.weight" not in new_state_dict
        and final_norm is not None
    ):
        new_state_dict["shared_head_norm.weight"] = final_norm
    return new_state_dict
