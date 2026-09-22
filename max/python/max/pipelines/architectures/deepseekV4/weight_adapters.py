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

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# DeepSeek-V4 checkpoints carry no ``model.`` prefix and name every module after
# the reference ``inference/model.py`` attributes, so the MAX modules are named
# to match and the only rename left is the per-block quantization scale:
# the checkpoint stores it as ``<proj>.scale``, MAX reads ``<proj>.weight_scale``.
#
# Note this is a suffix match on ``.scale`` and must stay one: the mHC mixing
# parameters (``hc_attn_scale``, ``hc_ffn_scale``, ``hc_head_scale``) are
# underscore-suffixed tensors of their own, not scales of a ``weight``.
_SCALE_SUFFIX = ".scale"
_MAX_SCALE_SUFFIX = ".weight_scale"


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for name, value in state_dict.items():
        # TODO: Support DSpark (commit 8). The ``mtp.*`` stages are dropped for
        # now, mirroring what the DeepSeek-V3 checkpoint converter does with its
        # own MTP layer.
        if name.startswith("mtp."):
            continue

        max_name = name
        if max_name.endswith(_SCALE_SUFFIX):
            max_name = max_name[: -len(_SCALE_SUFFIX)] + _MAX_SCALE_SUFFIX

        new_state_dict[max_name] = value.data()

    return new_state_dict
