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

from typing import cast

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.lfm2.weight_adapters import (
    convert_lfm2_safetensor_state_dict,
)


class _FakeWeights:
    def __init__(self, tensor: np.ndarray) -> None:
        self._tensor = tensor

    def data(self) -> WeightData:
        return WeightData.from_numpy(self._tensor, "fake")


def test_convert_lfm2_safetensor_state_dict_maps_attention_norm_names() -> None:
    state_dict = cast(
        dict[str, Weights],
        {
            "model.layers.0.self_attn.q_layernorm.weight": _FakeWeights(
                np.zeros((4,))
            ),
            "model.layers.0.self_attn.k_layernorm.weight": _FakeWeights(
                np.zeros((4,))
            ),
            "model.layers.0.self_attn.out_proj.weight": _FakeWeights(
                np.zeros((4, 4))
            ),
            "model.layers.0.conv.conv.weight": _FakeWeights(
                np.zeros((4, 1, 3))
            ),
        },
    )

    mapped = convert_lfm2_safetensor_state_dict(state_dict)
    assert "layers.0.self_attn.q_norm.weight" in mapped
    assert "layers.0.self_attn.k_norm.weight" in mapped
    assert "layers.0.self_attn.o_proj.weight" in mapped
    assert "layers.0.conv.conv_weight" in mapped
