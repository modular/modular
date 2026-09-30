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
from .moe import (
    MoE,
    MoEGate,
    make_concatenated_gated_activation_fn,
    make_interleaved_gated_activation_fn,
)
from .moe_fp8 import MoEQuantized
from .quant_strategy import (
    BlockScaledStrategy,
    Fp8Strategy,
    Mxfp6Strategy,
    Nvfp4Scales,
    NvMxf4f8Strategy,
    QuantStrategy,
    interleaved_block_scales_shape,
)
from .sigmoid_router import SigmoidTopKRouter
from .stacked_moe import (
    GateUpFormat,
    StackedMoE,
    make_stacked_gated_activation_fn,
)

__all__ = [
    "Fp8Strategy",
    "GateUpFormat",
    "MoE",
    "MoEGate",
    "MoEQuantized",
    "NvMxf4f8Strategy",
    "Nvfp4Scales",
    "QuantStrategy",
    "SigmoidTopKRouter",
    "StackedMoE",
    "forward_moe_sharded_layers",
    "interleaved_block_scales_shape",
    "make_concatenated_gated_activation_fn",
    "make_interleaved_gated_activation_fn",
    "make_stacked_gated_activation_fn",
]
