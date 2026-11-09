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
"""Assigns Inkling checkpoint weights to registered role namespaces."""

from __future__ import annotations

import re

from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

from ..inkling.weight_adapters import (
    VISION_PREFIX,
    convert_safetensor_state_dict,
)

_EXPERT_SCALE = re.compile(
    r"(\.experts\.w(?:13|2))_weight\.(weight_scale(?:_2)?|input_scale)$"
)


def convert_registered_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Reuses checkpoint conversion and preserves ModuleV3 expert scale names."""
    converted = convert_safetensor_state_dict(
        state_dict, huggingface_config, **unused_kwargs
    )
    return {
        (
            name
            if name.startswith(VISION_PREFIX)
            else "decoder." + _EXPERT_SCALE.sub(r"\1_\2", name)
        ): value
        for name, value in converted.items()
    }
