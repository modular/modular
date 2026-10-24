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
"""Quantization scheme of Xiaomi's MiMo-V2.6-Flash checkpoint.

The checkpoint's quantization does not differ between the graph-API and
ModuleV3 ports, so this package re-exports the graph-API architecture's
scheme. Only the expert-parallel dispatch quantization is defined here: it
is a ModuleV3 concern (the graph-API architecture has no EP path).
"""

from __future__ import annotations

from max.dtype import DType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from ..mimo_v2.quant import (  # re-export
    FP8_BLOCK,
    MXFP4_BLOCK,
    MiMoV2QuantScheme,
    moe_layers,
    parse_quant_scheme,
)

__all__ = [
    "FP8_BLOCK",
    "MXFP4_BLOCK",
    "MXFP8_DISPATCH_QUANT_CONFIG",
    "MiMoV2QuantScheme",
    "moe_layers",
    "parse_quant_scheme",
]

# The expert-parallel dispatch quantizes tokens to the routed experts'
# activation format: MXFP8, one E8M0 scale per 32 inputs. The experts'
# own config is MXFP4 (the weight format), which the dispatch kernels do
# not read -- they branch on this config's format, and MXFP8 selects the
# NVIDIA block-scaled dispatch layout whose received activations feed the
# W4A8 grouped matmuls unchanged.
MXFP8_DISPATCH_QUANT_CONFIG = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, MXFP4_BLOCK),
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, MXFP4_BLOCK),
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
    format=QuantFormat.MXFP8,
)
