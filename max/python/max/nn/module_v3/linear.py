# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""A Module for linear transformations."""

from __future__ import annotations

from typing import Literal

from ...experimental import random
from ...experimental.tensor import Tensor
from ...graph import Dim, DimLike
from .module import Module


class Linear(Module):
    """A unary linear transformation over an input tensor.

    Linear is defined as `f(x) = x @ W.T + B` where `W` is the
    weight tensor and B is an optional bias tensor.

    If W is not square then the transformation represents a
    dimensionality change. By convention the weight tensor is stored
    transposed.

    .. code-block:: python

        from max.nn.module_v3 import Linear
        from max.experimental.tensor import Tensor

        model = Linear(5, 10)

        assert dict(model.parameters) == {
            "weight": model.weight, "bias": model.bias
        }

        result = model(Tensor.ones([5]))
        assert result.shape == [10]
    """

    # By convention weight is stored transposed
    # ie. weight.shape == [out_dim, in_dim]
    weight: Tensor
    """The weight :obj:`~max.experimental.tensor.Tensor` for the linear transformation."""
    bias: Tensor | Literal[0]
    """The bias :obj:`~max.experimental.tensor.Tensor` for the linear transformation (or 0 if bias is disabled)."""

    def __init__(self, in_dim: DimLike, out_dim: DimLike, *, bias: bool = True):
        """Constructs a random linear transformation of the given dimensions.

        Args:
            in_dim: The dimensionality of the input to the transformation
            out_dim: The dimensionality after applying the transformation
                to the input tensor of dim `in_dim`.
            bias: Whether to use a `bias` in the transformation.
        """
        self.weight = random.normal([out_dim, in_dim])
        self.bias = random.normal([out_dim]) if bias else 0

    @property
    def in_dim(self) -> Dim:
        """The input dimension for the transformation."""
        return self.weight.shape[1]

    @property
    def out_dim(self) -> Dim:
        """The output dimension for the transformation."""
        return self.weight.shape[0]

    def __rich_repr__(self):
        """Repr matching the Linear constructor."""
        yield "in_dim", self.in_dim
        yield "out_dim", self.out_dim
        yield "bias", isinstance(self.bias, Tensor), True

    def __call__(self, x: Tensor) -> Tensor:
        """Applies a linear transformation to the input tensor.

        Linear is defined as `f(x) = x @ W.T + B` where `W` is the
        weight tensor and B is an optional bias tensor.

        Args:
            x: The input tensor
        Returns:
            The result of applying the linear transformation to the tensor.
        """
        return x @ self.weight.T + self.bias
