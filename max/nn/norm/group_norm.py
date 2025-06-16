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

from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops

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
        device: DeviceRef = DeviceRef.GPU(),
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

        gamma = (
            self.weight.cast(x.dtype).to(x.device)
            if self.affine and self.weight
            else ops.constant(
                np.full((self.num_channels,), 1.0, dtype=np.float32),
                dtype=x.dtype,
                device=DeviceRef.CPU(),
            ).to(x.device)
        )

        beta = (
            self.bias.cast(x.dtype).to(x.device)
            if self.affine and self.bias
            else ops.constant(
                np.full((self.num_channels,), 0.0, dtype=np.float32),
                dtype=x.dtype,
                device=DeviceRef.CPU(),
            ).to(x.device)
        )

        return ops.custom(
            "group_norm",
            x.device,
            [
                x,
                gamma,
                beta,
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.num_groups, dtype=DType.int32, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        )[0].tensor
