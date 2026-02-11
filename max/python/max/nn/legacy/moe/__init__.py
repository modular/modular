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
"""Mixture of Experts (MoE) module."""

from .moe import MoE, MoEGate
from .moe_fp8 import MoEQuantized
from .quant_strategy import (
    Fp8Strategy,
    Nvfp4Scales,
    Nvfp4Strategy,
    QuantStrategy,
    silu_gate,
)
from .stacked_moe import GateUpFormat, StackedMoE, silu_activation

__all__ = [
    "Fp8Strategy",
    "GateUpFormat",
    "MoE",
    "MoEGate",
    "MoEQuantized",
    "Nvfp4Scales",
    "Nvfp4Strategy",
    "QuantStrategy",
    "StackedMoE",
    "silu_activation",
    "silu_gate",
]
