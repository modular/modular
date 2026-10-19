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
"""Maps the Hy4 checkpoint onto the unified MTP module's names."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.driver import Buffer, bfloat16_as_uint16_to_float32
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

from ..hy_v4.weight_adapters import (
    convert_mtp_state_dict,
    convert_safetensor_state_dict,
)

# BF16 in the MTP layer, float32 in every target layer and in the modules.
DRAFT_FLOAT32 = (
    "draft.self_attn.learnable_sink_param",
    "draft.mlp.gate.e_score_correction_bias",
)


def _bf16_to_float32(weight: WeightData) -> WeightData:
    if weight.dtype != DType.bfloat16:
        raise ValueError(
            f"{weight.name}: expected bfloat16, got {weight.dtype}"
        )
    bits = np.from_dlpack(weight.to_buffer().view(DType.uint16))
    values = bfloat16_as_uint16_to_float32(bits.view(np.uint16))
    buffer = Buffer.from_numpy(np.ascontiguousarray(values))
    return WeightData(buffer, weight.name, DType.float32, Shape(buffer.shape))


def convert_with_mtp_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Returns the target's ``language_model.*`` and the draft's ``draft.*``.

    The draft's BF16 sink and router bias are widened to the float32 the
    target layers store, which is exact.
    """
    out = convert_safetensor_state_dict(state_dict, huggingface_config)
    draft = convert_mtp_state_dict(state_dict, huggingface_config)
    for name in DRAFT_FLOAT32:
        draft[name] = _bf16_to_float32(draft[name])
    return out | draft
