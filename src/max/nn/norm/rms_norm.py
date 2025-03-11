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

from ..layer import Layer, Module


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


class RMSNormV2(Module):
    """Computes the Root Mean Square normalization on inputs.

    Args:
        dim: Size of last dimension of the expected input.
        eps: Value added to denominator for numerical stability.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = Weight("weight", DType.float32, [dim])
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        weight: TensorValue = ops.cast(self.weight, x.dtype)
        if x.device:
            weight = weight.to(x.device)
        return ops.custom(
            "rms_norm",
            [x, weight, ops.cast(self.eps, x.dtype)],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        )[0].tensor


class DistributedRMSNorm(RMSNormV2):
    def __init__(self, *args, devices: list[DeviceRef], **kwargs):
        super().__init__(*args, **kwargs)
        self.num_devices = len(devices)

        clone_weight = lambda weight, i: weight
        self.weight.set_sharding_strategy(clone_weight)
        # Create a separate RMS layer for each device.
        self.rms_norms = []
        for n, device in enumerate(devices):
            layer = RMSNormV2(*args, **kwargs)
            layer.weight = self.weight.shard(n, device)
            self.rms_norms.append(layer)

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:  # type: ignore[override]
        return [self.rms_norms[i](xs[i]) for i in range(self.num_devices)]
