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
"""Mistral Large 4 gated MLP, in the checkpoint's own projection names.

The layer is ``w2(silu(w1(x)) * w3(x))`` -- gate, up, down in the checkpoint's
numbering -- and it is the one the dense layer, the shared expert and every
routed expert use. The quantization differs per site: layer 0's FFN and the
shared expert are FP8 block-scaled, the routed experts are NVFP4, and a BF16
checkpoint quantizes none of them, so the config comes in from outside.

It subclasses :class:`QuantizedMLP`, whose computation it is, because the
expert-parallel MoE machinery is written against that type and because a
TF ``ModuleList`` of experts has to be typed as one. What it overrides is the
naming: the three projections are ``w1``/``w3``/``w2``, which is what the
weight contract (and therefore Mach's loader) calls them, so the inherited
``gate_proj``/``up_proj``/``down_proj`` names are never built and the module
tree holds only the checkpoint's three.
"""

from __future__ import annotations

from max import functional as F
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_linear import QuantizedLinear, QuantizedMLP
from max.nn.legacy.quant_config import QuantConfig
from max.tensor import Tensor


class MistralLarge4MLP(QuantizedMLP):
    """``w2(silu(w1(x)) * w3(x))``."""

    w1: QuantizedLinear
    """The gate projection."""
    w3: QuantizedLinear
    """The up projection."""
    w2: QuantizedLinear
    """The down projection."""

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Builds the three projections.

        Args:
            hidden_dim: Width of the input and output.
            feed_forward_length: Width of the hidden FFN.
            quant_config: The site's quantization, or ``None`` for BF16.
        """
        # Deliberately not ``super().__init__``: QuantizedMLP's constructor
        # builds DeepSeek's three projection names, and this layer must hold
        # the checkpoint's three instead.
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length
        self.w1 = QuantizedLinear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=False,
            quant_config=quant_config,
        )
        self.w3 = QuantizedLinear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=False,
            quant_config=quant_config,
        )
        self.w2 = QuantizedLinear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            bias=False,
            quant_config=quant_config,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs the gated MLP over ``x``."""
        # One matmul over the concatenated gate and up rows: the two weight
        # tiles stack exactly, block scales included.
        gate_up = quant_ops.matmul(
            x, quant_ops.concat_weights(self.w1.weight, self.w3.weight, axis=0)
        )
        # The two projections stack into one output, so the split is on its own
        # width rather than on ``feed_forward_length``: a mesh-mapped tensor
        # carries the model's width, but the local shard the shared expert's
        # side stream hands this layer is one rank's slice.
        half = int(gate_up.shape[-1]) // 2
        gate, up = gate_up.split([half, half], axis=-1)
        return self.w2(F.silu(gate) * up)
