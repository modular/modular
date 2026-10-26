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
"""Maps a Hy4 checkpoint's MTP layer onto the draft module's names."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.driver import Buffer, bfloat16_as_uint16_to_float32
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

from ..hy_v4.weight_adapters import convert_mtp_state_dict

__all__ = ["convert_hy_v4_mtp_state_dict"]

_DRAFT_PREFIX = "draft."

# BF16 in the MTP layer, float32 in every target layer and in the modules.
_DRAFT_FLOAT32 = (
    "self_attn.learnable_sink_param",
    "mlp.gate.e_score_correction_bias",
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


def convert_hy_v4_mtp_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Returns the MTP layer's own tensors, named as the draft module's.

    Its BF16 sink and router bias are widened to the float32 the target
    layers store, which is exact.
    """
    draft = {
        name.removeprefix(_DRAFT_PREFIX): weight
        for name, weight in convert_mtp_state_dict(
            state_dict, huggingface_config
        ).items()
    }
    for name in _DRAFT_FLOAT32:
        draft[name] = _bf16_to_float32(draft[name])
    return draft
