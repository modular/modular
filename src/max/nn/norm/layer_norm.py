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

"""Layer Normalization layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops

from ..layer import Layer, Module


@dataclass
class LayerNormV1(Layer):
    """Layer normalization block.

    Deprecated: Use `LayerNorm` instead.
    """

    weight: TensorValue
    bias: TensorValue | None = None
    eps: float = 1e-6

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        bias: Any = (
            ops.cast(self.bias, input.dtype)
            if self.bias
            # If bias wasn't passed then use bias-less layer norm (beta = 0).
            else ops.broadcast_to(
                ops.constant(0.0, input.dtype, device=DeviceRef.CPU()),
                shape=(input.shape[-1],),
            )
        )

        weight: Any = self.weight
        if weight.type.device != input.type.device:
            weight = weight.to(input.type.device or DeviceRef.CPU())

        if bias and bias.type.device != input.type.device:
            bias = bias.to(input.type.device or DeviceRef.CPU())

        res = ops.layer_norm(
            input,
            gamma=ops.cast(weight, input.dtype),
            beta=bias,
            epsilon=self.eps,
        )
        return res


class LayerNorm(Module):
    """Layer normalization block."""

    def __init__(
        self, dims: int, device: DeviceRef, eps: float = 1e-5, use_bias=True
    ):
        self.weight = Weight("weight", DType.float32, (dims,), device=device)
        self.bias = (
            Weight("bias", DType.float32, (dims,), device=device)
            if use_bias
            else None
        )
        self.eps = eps

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        bias = (
            ops.cast(self.bias, input.dtype)
            if self.bias
            # If bias wasn't passed then use bias-less layer norm (beta = 0).
            else ops.broadcast_to(
                ops.constant(0.0, input.dtype, self.weight.device),
                shape=(input.shape[-1],),
            )
        )
        return ops.layer_norm(
            input,
            gamma=ops.cast(self.weight, input.dtype),
            beta=bias,
            epsilon=self.eps,
        )
