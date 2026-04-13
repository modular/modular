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
"""Weight adaptation utilities for FLUX2 transformer models.

Handles NVFP4 checkpoint conversion and stacked QKV splitting so that
both the ComponentModel and PipelineExecutor paths share the same logic.
"""

from __future__ import annotations

from max.graph.weights import WeightData
from max.nn.quant_config import QuantConfig

from ..flux2_modulev3.model import Flux2TransformerModel as _V3TransformerModel
from ..flux2_modulev3.nvfp4_weight_adapter import convert_nvfp4_state_dict

_split_stacked_qkv = _V3TransformerModel._split_stacked_qkv


def adapt_weights(
    state_dict: dict[str, WeightData],
    quant_config: QuantConfig | None = None,
) -> dict[str, WeightData]:
    """Apply NVFP4 conversion and QKV splitting to a raw state dict.

    Args:
        state_dict: Raw checkpoint weights keyed by parameter name.
        quant_config: If not None, apply NVFP4 weight conversion.

    Returns:
        Adapted state dict with BFL naming converted and stacked QKV
        weights split into separate Q, K, V entries.
    """
    if quant_config is not None:
        state_dict = convert_nvfp4_state_dict(state_dict)

    stacked_qkv = any(
        ".attn.qkv_proj." in k or ".attn.add_qkv_proj." in k for k in state_dict
    )
    if stacked_qkv:
        state_dict = _split_stacked_qkv(state_dict)

    return state_dict
