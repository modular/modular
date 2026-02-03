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
"""Activation function modules for MAX neural networks."""

from max import functional as F
from max.tensor import Tensor

from .module import Module
from .linear import Linear


class SiLU(Module[[Tensor], Tensor]):
    """Sigmoid Linear Unit (SiLU/Swish) activation function module.

    Applies the SiLU activation function element-wise:
        SiLU(x) = x * sigmoid(x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply SiLU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with SiLU applied element-wise.
        """
        return F.silu(x)


class Identity(Module[[Tensor], Tensor]):
    """Identity module that returns the input unchanged.

    Useful as a placeholder or when conditionally disabling layers.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return input unchanged.

        Args:
            x: Input tensor.

        Returns:
            The same input tensor.
        """
        return x


class Dropout(Module[[Tensor], Tensor]):
    """Dropout module for regularization.

    During inference (which is the primary use case in MAX), this module
    acts as a pass-through and returns the input unchanged.

    Args:
        p: Probability of an element to be zeroed during training.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize Dropout module.

        Args:
            p: Dropout probability (unused during inference).
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout (no-op during inference).

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged during inference.
        """
        # During inference, dropout is a no-op
        return x


class GELU(Module[[Tensor], Tensor]):
    """GELU activation function with optional tanh approximation.

    Applies the Gaussian Error Linear Unit activation function.
    When approximate="tanh", uses the faster tanh approximation.

    Args:
        dim_in: The number of channels in the input.
        dim_out: The number of channels in the output.
        approximate: If "tanh", use tanh approximation. Defaults to "none".
        bias: Whether to use a bias in the linear layer. Defaults to True.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply linear projection followed by GELU activation.

        Args:
            hidden_states: Input tensor.

        Returns:
            Tensor with linear projection and GELU applied.
        """
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class GEGLU(Module[[Tensor], Tensor]):
    """GEGLU activation function, a gated linear unit variant.

    A variant of the gated linear unit that uses GELU for gating.
    See: https://huggingface.co/papers/2002.05202

    Args:
        dim_in: The number of channels in the input.
        dim_out: The number of channels in the output.
        bias: Whether to use a bias in the linear layer. Defaults to True.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply GEGLU activation.

        Args:
            hidden_states: Input tensor.

        Returns:
            Tensor with GEGLU applied.
        """
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class SwiGLU(Module[[Tensor], Tensor]):
    """SwiGLU activation function, a gated linear unit variant.

    Similar to GEGLU but uses SiLU/Swish instead of GELU for gating.
    See: https://huggingface.co/papers/2002.05202

    Args:
        dim_in: The number of channels in the input.
        dim_out: The number of channels in the output.
        bias: Whether to use a bias in the linear layer. Defaults to True.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply SwiGLU activation.

        Args:
            hidden_states: Input tensor.

        Returns:
            Tensor with SwiGLU applied.
        """
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class ApproximateGELU(Module[[Tensor], Tensor]):
    """Approximate GELU activation using sigmoid approximation.

    Uses the approximation: x * sigmoid(1.702 * x)
    See section 2 of: https://huggingface.co/papers/1606.08415

    Args:
        dim_in: The number of channels in the input.
        dim_out: The number of channels in the output.
        bias: Whether to use a bias in the linear layer. Defaults to True.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply approximate GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with approximate GELU applied.
        """
        x = self.proj(x)
        return x * F.sigmoid(1.702 * x)


class FeedForward(Module[[Tensor], Tensor]):
    """Configurable feed-forward network with various activation functions.

    A feed-forward layer commonly used in transformers with support for
    different gated activation functions.

    Args:
        dim: The number of channels in the input.
        dim_out: The number of channels in the output. Defaults to dim.
        mult: The multiplier for the hidden dimension. Defaults to 4.
        dropout: The dropout probability. Defaults to 0.0.
        activation_fn: Activation function name. One of "gelu", "gelu-approximate",
            "geglu", "geglu-approximate", "swiglu". Defaults to "geglu".
        final_dropout: Whether to apply dropout after the output. Defaults to False.
        inner_dim: Override for hidden dimension. Defaults to dim * mult.
        bias: Whether to use bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        else:
            raise ValueError(f"Unknown activation_fn: {activation_fn}")

        from .sequential import ModuleList

        self.net = ModuleList([])
        self.net.append(act_fn)
        self.net.append(Dropout(dropout))
        self.net.append(Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(Dropout(dropout))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward network.

        Args:
            hidden_states: Input tensor.

        Returns:
            Tensor after feed-forward transformation.
        """
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
