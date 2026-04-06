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

"""Group Normalization implementation using the graph API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops

from ..layer import Module


@dataclass
class GroupNorm(Module):
    """Group normalization block.

    This layer divides channels into groups and computes normalization
    statistics per group.

    When called, ``GroupNorm`` accepts a :class:`~max.graph.TensorValue` of shape
    ``(N, C, *)`` where ``C`` is the number of channels. Then, it returns a
    normalized :class:`~max.graph.TensorValue` of the same shape.

    Args:
        num_groups: The number of groups to divide the channels into.
        num_channels: The number of input channels.
        eps: A small value added to the denominator for numerical stability.
        affine: Whether to apply a learnable affine transformation after
            normalization.
        device: The target :class:`~max.graph.DeviceRef` for computation.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        io_layout: Literal["nchw", "nhwc"] = "nchw",
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.io_layout = io_layout

        if self.io_layout not in ("nchw", "nhwc"):
            raise ValueError(
                "io_layout must be one of 'nchw' or 'nhwc'. "
                f"Got {self.io_layout!r}."
            )

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
            x: Input tensor. ``io_layout="nchw"`` expects shape ``[N, C, *]``.
                ``io_layout="nhwc"`` expects rank-4 shape ``[N, H, W, C]``.

        Returns:
            Normalized tensor of same shape as input
        """
        # Input shape validation.
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with >=2 dimensions, got shape {x.shape}"
            )

        if self.io_layout == "nhwc" and x.rank != 4:
            raise ValueError(
                "io_layout='nhwc' requires a rank-4 input tensor, got "
                f"shape {x.shape}"
            )

        channel_axis = 1 if self.io_layout == "nchw" else x.rank - 1
        if x.shape[channel_axis] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        input_x = (
            x if self.io_layout == "nchw" else ops.permute(x, [0, 3, 1, 2])
        )

        gamma = (
            self.weight.cast(input_x.dtype).to(input_x.device)
            if self.affine and self.weight
            else ops.constant(
                np.full((self.num_channels,), 1.0, dtype=np.float32),
                dtype=input_x.dtype,
                device=DeviceRef.CPU(),
            ).to(input_x.device)
        )

        beta = (
            self.bias.cast(input_x.dtype).to(input_x.device)
            if self.affine and self.bias
            else ops.constant(
                np.full((self.num_channels,), 0.0, dtype=np.float32),
                dtype=input_x.dtype,
                device=DeviceRef.CPU(),
            ).to(input_x.device)
        )

        output = ops.custom(
            "group_norm",
            input_x.device,
            [
                input_x,
                gamma,
                beta,
                ops.constant(
                    self.eps, dtype=input_x.dtype, device=DeviceRef.CPU()
                ),
                ops.constant(
                    self.num_groups, dtype=DType.int32, device=DeviceRef.CPU()
                ),
            ],
            [
                TensorType(
                    dtype=input_x.dtype,
                    shape=input_x.shape,
                    device=input_x.device,
                )
            ],
        )[0].tensor

        if self.io_layout == "nhwc":
            output = ops.permute(output, [0, 2, 3, 1])
        return output
