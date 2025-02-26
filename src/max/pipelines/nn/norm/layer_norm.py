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

from max.dtype import DType
from max.graph import TensorValue, Weight, ops

from ..layer import Layer, LayerV2


@dataclass
class LayerNorm(Layer):
    """Layer normalization block."""

    weight: Weight
    bias: Weight | None = None
    eps: float = 1e-6

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        bias = (
            ops.cast(self.bias, input.dtype)
            if self.bias
            # If bias wasn't passed then use bias-less layer norm (beta = 0).
            else ops.broadcast_to(
                ops.constant(0.0, input.dtype), shape=(input.shape[-1],)
            )
        )
        return ops.layer_norm(
            input,
            gamma=ops.cast(self.weight, input.dtype),
            beta=bias,
            epsilon=self.eps,
        )


class LayerNormV2(LayerV2):
    """Layer normalization block."""

    def __init__(self, dims: int, eps: float = 1e-5, use_bias=True):
        self.weight = Weight("weight", DType.float32, (dims,))
        self.bias = Weight("bias", DType.float32, (dims,)) if use_bias else None
        self.eps = eps

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        bias = (
            ops.cast(self.bias, input.dtype)
            if self.bias
            # If bias wasn't passed then use bias-less layer norm (beta = 0).
            else ops.broadcast_to(
                ops.constant(0.0, input.dtype), shape=(input.shape[-1],)
            )
        )
        return ops.layer_norm(
            input,
            gamma=ops.cast(self.weight, input.dtype),
            beta=bias,
            epsilon=self.eps,
        )
