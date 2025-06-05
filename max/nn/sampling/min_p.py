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
"""MinP Sampler custom ops."""

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, Shape, ShapeLike, TensorType, TensorValue, ops


class MinPSampler(nn.Module):
    """A min_p sampler."""

    dtype: DType
    shape: Shape
    min_p: float
    temperature: float

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        min_p: float = 0.0,
        temperature: float = 1,
    ):
        self.dtype = dtype
        self.shape = Shape(shape)
        self.min_p = min_p
        self.temperature = temperature

    def __call__(self, input: TensorValue) -> TensorValue:
        return ops.custom(
            "min_p_sampling",
            input.device,
            [
                ops.constant(
                    self.min_p, dtype=self.dtype, device=DeviceRef.CPU()
                ),
                input,
                ops.constant(
                    self.temperature, dtype=self.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(self.dtype, self.shape, device=input.device)],
        )[0].tensor
