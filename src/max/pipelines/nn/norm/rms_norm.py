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

"""Normalization layer."""

from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    DeviceRef,
    TensorType,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)

from ..layer import Layer, LayerV2


@dataclass
class RMSNorm(Layer):
    weight: TensorValueLike
    eps: float = 1e-6

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.custom(
            "rms_norm",
            [x, ops.cast(self.weight, x.dtype), ops.cast(self.eps, x.dtype)],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        )[0].tensor


@dataclass
class DistributedRMSNorm(Layer):
    rms_norms: list[RMSNorm]
    devices: list[DeviceRef]

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:
        return [self.rms_norms[i](xs[i]) for i in range(len(self.devices))]


class RMSNormV2(LayerV2):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Weight("weight", DType.float32, [dim])
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.custom(
            "rms_norm",
            [x, ops.cast(self.weight, x.dtype), ops.cast(self.eps, x.dtype)],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        )[0].tensor
