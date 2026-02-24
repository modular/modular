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

"""Group normalization."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3.module import Module


def group_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    eps: float,
) -> Tensor:
    """Applies Group Normalization to an input tensor.

    Group Normalization divides the channels into groups and computes
    normalization statistics within each group. This is useful for small
    batch sizes where batch normalization is unstable.

    Args:
        x: Input tensor of shape [N, C, \\*] where C is number of channels
        weight: Weight tensor of shape [C]
        bias: Bias tensor of shape [C]
        num_groups: Number of groups to separate the channels into
        eps: Small constant added to denominator for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    if len(x.shape) < 2:
        raise ValueError(
            f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
        )

    return F.custom(
        "group_norm",
        x.device,
        [
            x,
            weight.to(x.device),
            bias.to(x.device),
            F.constant(eps, dtype=x.dtype, device=CPU()),
            F.constant(num_groups, dtype=DType.int32, device=CPU()),
        ],
        [x.type],
    )[0]


class GroupNorm(Module[[Tensor], Tensor]):
    """Group normalization block.

    Divides channels into groups and computes normalization stats per group.
    Follows the implementation pattern from PyTorch's group_norm.

    This implementation uses Tensor instead of Weight, which
    automatically handles dtype matching with input tensors, eliminating
    the need for dtype workarounds.

    Example:
        .. code-block:: python

            from max.nn.module_v3 import GroupNorm
            from max.experimental.tensor import Tensor

            norm = GroupNorm(num_groups=32, num_channels=128)
            x = Tensor.ones([1, 128, 32, 32])
            result = norm(x)
    """

    weight: Tensor | None
    """The weight tensor with shape [num_channels] (None if affine=False)."""
    bias: Tensor | None
    """The bias tensor with shape [num_channels] (None if affine=False)."""
    num_groups: int
    """Number of groups to separate the channels into."""
    num_channels: int
    """Number of input channels."""
    eps: float
    """Small constant added to denominator for numerical stability."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        """Initialize GroupNorm module.

        Args:
            num_groups: Number of groups to separate the channels into
            num_channels: Number of input channels
            eps: Small constant added to denominator for numerical stability.
                Default: 1e-5
            affine: If True, apply learnable affine transform parameters.
                Default: True
        """
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels({num_channels}) should be divisible by "
                f"num_groups({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = Tensor.ones([num_channels])
            self.bias = Tensor.zeros([num_channels])
        else:
            self.weight = None
            self.bias = None

    def __rich_repr__(self):
        """Rich representation for debugging."""
        yield "num_groups", self.num_groups
        yield "num_channels", self.num_channels
        yield "eps", self.eps, 1e-5
        yield "affine", self.affine, True

    def forward(self, x: Tensor) -> Tensor:
        """Apply group normalization to input tensor.

        Args:
            x: Input tensor of shape [N, C, \\*] where C is number of channels

        Returns:
            Normalized tensor of same shape as input
        """
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        if self.affine:
            if self.weight is None or self.bias is None:
                raise ValueError("weight and bias must be set when affine=True")
            weight = self.weight
            bias = self.bias
        else:
            # Create temporary tensors of ones and zeros when affine=False
            weight = Tensor.ones(
                [self.num_channels], dtype=x.dtype, device=x.device
            )
            bias = Tensor.zeros(
                [self.num_channels], dtype=x.dtype, device=x.device
            )

        return group_norm(x, weight, bias, self.num_groups, self.eps)
