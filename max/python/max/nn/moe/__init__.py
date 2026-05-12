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

from .expert_parallel import forward_moe_sharded_layers
from .moe import MoE, MoEGate
from .moe_fp8 import MoEQuantized
from .quant_strategy import (
    Fp8Strategy,
    Mxfp4Strategy,
    Nvfp4Scales,
    Nvfp4Strategy,
    QuantStrategy,
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
    "forward_moe_sharded_layers",
    "silu_activation",
]
