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

"""Quantize-aware Linear / MLP layers."""

from __future__ import annotations

from typing import Literal

from max.experimental.nn import Module
from max.experimental.nn.common_layers.activation import (
    activation_function_from_name,
)
from max.experimental.tensor import Tensor
from max.nn.quant_config import QuantConfig

from . import quant_ops
from .quant_ops import QuantAwareTensor


class QuantizedLinear(Module[[Tensor], Tensor]):
    """Quantize-aware linear transformation ``x @ weight.T + bias``."""

    weight: QuantAwareTensor
    bias: Tensor | Literal[0]
    """Bias :obj:`~max.experimental.tensor.Tensor`, or ``0`` when disabled."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = quant_ops.quantized_weight(out_dim, in_dim, quant_config)
        self.bias = Tensor.zeros((out_dim,)) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        return quant_ops.matmul(x, self.weight) + self.bias


class QuantizedMLP(Module[[Tensor], Tensor]):
    """Quantize-aware gated MLP.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))`` with all three
    projections sharing the ``quant_config`` (bf16 or FP8 block-scaled).
    """

    gate_proj: QuantizedLinear
    up_proj: QuantizedLinear
    down_proj: QuantizedLinear

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
        bias: bool = False,
        activation_function: str = "silu",
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length
        self.gate_proj = QuantizedLinear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=bias,
            quant_config=quant_config,
        )
        self.up_proj = QuantizedLinear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=bias,
            quant_config=quant_config,
        )
        self.down_proj = QuantizedLinear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            bias=bias,
            quant_config=quant_config,
        )
        self.activation_function = activation_function_from_name(
            activation_function
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation_function(gate) * up)
