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

The pipeline loads the checkpoint through
:func:`~.weight_rules.mimo_v2_weight_rules`.
:func:`convert_safetensor_state_dict` applies the same rules to a mapping of
weights, for the tests that build the decoder from a checkpoint by hand.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from max.graph.weights import WeightData, Weights
from max.pipelines.lib.weight_transform import WeightTransform

from .weight_rules import mimo_v2_weight_rules


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Renames the checkpoint's text-model tensors to MAX's MiMo-V2 weights.

    Args:
        state_dict: Checkpoint tensors, by checkpoint name.
        **unused_kwargs: Accepted so every adapter shares one signature.

    Returns:
        The weights :func:`~.weight_rules.mimo_v2_weight_rules` keeps, keyed
        and named by MAX name.
    """
    adapted = WeightTransform(mimo_v2_weight_rules())(state_dict)
    return {
        name: dataclasses.replace(data, name=name)
        for name, data in adapted.items()
    }
