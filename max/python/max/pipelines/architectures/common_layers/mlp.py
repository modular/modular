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

"""A gated MLP layer (ModuleV3)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from .activation import activation_function_from_name


class MLP(Module[[Tensor], Tensor]):
    """Simple multi-layer perceptron composed of three Linear layers.

    Computes the gated MLP transformation as::

        down_proj(activation_function(gate_proj(x)) * up_proj(x))

    The gate and up projections are fused into a single matmul for efficiency.
    Defaults to SiLU activation function.
    """

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
        bias: bool = False,
        activation_function: str = "silu",
    ) -> None:
        """Initializes the MLP layer.

        Args:
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of dimension used to project the inputs.
            bias: Whether to include bias terms in the linear layers.
            activation_function: Activation function to use. Options are:

                - ``silu``
                - ``gelu``
                - ``gelu_tanh``
                - ``relu``
                - ``tanh``
                - ``sigmoid``

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length
        self.has_bias = bias
        self.gate_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=bias,
        )
        self.down_proj = Linear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            bias=bias,
        )
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=bias,
        )
        self.activation_function = activation_function_from_name(
            activation_function
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies the MLP transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the MLP layers.
        """
        # Optimization to compute a single matmul by merging the
        # gate and up projection weights.
        feed_forward_length = int(self.gate_proj.weight.shape[0])
        gate_proj_weight: Tensor = self.gate_proj.weight.to(x.device)
        up_proj_weight: Tensor = self.up_proj.weight.to(x.device)

        if self.has_bias:
            assert isinstance(self.gate_proj.bias, Tensor)
            assert isinstance(self.up_proj.bias, Tensor)
            gate_proj_bias: Tensor = self.gate_proj.bias.to(x.device)
            up_proj_bias: Tensor = self.up_proj.bias.to(x.device)
            bias = F.concat((gate_proj_bias, up_proj_bias))

            output = (x @ F.concat((gate_proj_weight, up_proj_weight)).T) + bias
        else:
            output = x @ F.concat((gate_proj_weight, up_proj_weight)).T

        gate_out, up_out = F.split(
            output, [feed_forward_length, feed_forward_length], axis=1
        )
        assert isinstance(gate_out, Tensor)
        assert isinstance(up_out, Tensor)

        hidden = self.activation_function(gate_out) * up_out
        return self.down_proj(hidden)
