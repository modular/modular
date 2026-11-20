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
"""Tests the weight adapter the Eagle3 MLA draft architecture registers."""

from __future__ import annotations

from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, WeightsFormat
from max.pipelines.architectures.eagle3_deepseekV3 import (
    eagle3_deepseekV3_draft_arch,
)


def _weight(name: str, dtype: type[np.generic]) -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = WeightData.from_numpy(
        np.ones((4, 4), dtype=dtype), name
    )
    return weight


def test_float32_draft_loads_as_bfloat16() -> None:
    # Revisions of nvidia/Kimi-K2.6-Eagle3 before 2026-06-11 ship float32,
    # and the head is built in bfloat16.
    adapter = eagle3_deepseekV3_draft_arch.weight_adapters[
        WeightsFormat.safetensors
    ]
    converted = adapter(
        {
            "fc.weight": _weight("fc.weight", np.float32),
            "layers.0.self_attn.o_proj.weight": _weight(
                "layers.0.self_attn.o_proj.weight", np.float32
            ),
            "layers.0.self_attn.o_proj.weight_scale": _weight(
                "layers.0.self_attn.o_proj.weight_scale", np.float32
            ),
        }
    )

    assert converted["fc.weight"].dtype == DType.bfloat16
    assert (
        converted["decoder_layer.self_attn.o_proj.weight"].dtype
        == DType.bfloat16
    )
    assert (
        converted["decoder_layer.self_attn.o_proj.weight_scale"].dtype
        == DType.float32
    )
