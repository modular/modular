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

"""Group Normalization implementation using the graph API."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops

from ..layer import Module


@dataclass
class GroupNorm(Module):
    """Group normalization block.

    Divides channels into groups and computes normalization stats per group.
    Follows the implementation pattern from PyTorch's group_norm.

    Args:
        num_groups: Number of groups to separate the channels into
        num_channels: Number of input channels
        eps: Small constant added to denominator for numerical stability
        affine: If True, apply learnable affine transform parameters
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.num_channels % self.num_groups != 0:
            raise ValueError(
                f"num_channels({self.num_channels}) should be divisible by "
                f"num_groups({self.num_groups})"
            )

        self.weight: Weight | None = None
        self.bias: Weight | None = None
        if self.affine:
            # Create affine parameters
            self.weight = Weight(
                name="weight",
                shape=(self.num_channels,),
                dtype=DType.float32,
                device=device,
            )
            self.bias = Weight(
                name="bias",
                shape=(self.num_channels,),
                dtype=DType.float32,
                device=device,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply group normalization to input tensor.

        Args:
            x: Input tensor of shape [N, C, *] where C is number of channels

        Returns:
            Normalized tensor of same shape as input
        """
        # Input shape validation.
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        input_shape = x.shape
        N = input_shape[0]
        C = input_shape[1]
        HW = math.prod(input_shape[2:])  # Product of remaining dimensions.
        G = self.num_groups

        # Reshape to [N, G, C/G, HW] for normalization
        x = ops.reshape(x, [N, G, C // G, HW])

        # Compute mean and variance over C/G and HW dimensions.
        # Keep dims for broadcasting.
        x_reshaped = x.reshape((N, G, 1, -1))
        mean = ops.mean(x_reshaped)
        n_elements = (
            ops.shape_to_tensor((C // G * HW,)).cast(x.dtype).to(x.device)
        )
        var = ops.sum((x_reshaped - mean) ** 2) / n_elements

        # Normalize
        x = (x - mean) / ops.sqrt(var + self.eps)

        # Reshape back to original shape
        x = ops.reshape(x, [N, G, -1])
        x = ops.reshape(x, [N, C] + list(input_shape[2:]))  # Original shape

        # Apply affine transform if enabled
        if self.affine:
            assert self.weight is not None and self.bias is not None
            weight = ops.reshape(
                self.weight, [1, -1] + [1] * (len(x.shape) - 2)
            ).to(x.device)
            bias = ops.reshape(
                self.bias, [1, -1] + [1] * (len(x.shape) - 2)
            ).to(x.device)
            x = x * weight + bias

        return x
