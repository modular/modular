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
"""Deprecated alias of :mod:`max.nn.quant_config`."""

from max.nn.quant_config import (
    MXFP4_BLOCK_SIZE,
    NVFP4_BLOCK_SIZE,
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
    WeightSource,
    ceildiv,
    fp4_packed_k,
)

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "NVFP4_BLOCK_SIZE",
    "InputScaleSpec",
    "QuantConfig",
    "QuantFormat",
    "ScaleGranularity",
    "ScaleOrigin",
    "WeightScaleSpec",
    "WeightSource",
    "ceildiv",
    "fp4_packed_k",
]
