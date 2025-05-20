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
class RMSNormV1(Layer):
    """Computes the Root Mean Square normalization on inputs.

    Deprecated: Use `RMSNorm` instead.
    """

    weight: TensorValueLike
    eps: float = 1e-6
    weight_offset: float = 0.0
    multiply_before_cast: bool = True

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.custom(
            "rms_norm",
            [
                x,
                TensorValue(self.weight).cast(x.dtype),
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor


class RMSNorm(Module):
    """Computes the Root Mean Square normalization on inputs.

    Args:
        dim: Size of last dimension of the expected input.
        eps: Value added to denominator for numerical stability.
        weight_offset: Constant offset added to the learned weights at runtime.
            For Gemma-style RMSNorm, this should be set to 1.0.
        multiply_before_cast: True if we multiply the inputs by the learned
            weights before casting to the input type (Gemma3-style). False if we
            cast the inputs to the input type first, then multiply by the learned
            weights (Llama-style).
    """

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        weight_offset: float = 0.0,
        multiply_before_cast: bool = True,
    ):
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())
        self.eps = eps
        self.weight_offset = weight_offset
        self.multiply_before_cast = multiply_before_cast

    def __call__(self, x: TensorValue) -> TensorValue:
        weight: TensorValue = self.weight.cast(x.dtype)
        if x.device:
            weight = weight.to(x.device)

        return ops.custom(
            "rms_norm",
            [
                x,
                weight,
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor


class DistributedRMSNorm(RMSNorm):
    def __init__(self, *args, devices: list[DeviceRef], **kwargs):
        super().__init__(*args, **kwargs)
        self.num_devices = len(devices)

        clone_weight = lambda weight, i: weight
        self.weight.set_sharding_strategy(clone_weight)
        # Create a separate RMS layer for each device.
        self.rms_norms = []
        for n, device in enumerate(devices):
            layer = RMSNorm(*args, **kwargs)
            layer.weight = self.weight.shard(n, device)
            self.rms_norms.append(layer)

    def __call__(self, xs: list[TensorValue]) -> list[TensorValue]:  # type: ignore[override]
        return [self.rms_norms[i](xs[i]) for i in range(self.num_devices)]
