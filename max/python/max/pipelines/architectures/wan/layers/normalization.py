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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module


class WanLayerNorm(Module):
    """LayerNorm using the built-in fused ``ops.layer_norm`` kernel."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        use_bias: bool = True,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.has_weight = elementwise_affine
        self.has_bias = elementwise_affine and use_bias
        if elementwise_affine:
            self.weight = Weight("weight", dtype, [dim], device)
            if use_bias:
                self.bias = Weight("bias", dtype, [dim], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.has_weight and self.has_bias:
            return ops.layer_norm(x, self.weight, self.bias, epsilon=self.eps)
        if self.has_weight:
            return ops.layer_norm(x, self.weight, epsilon=self.eps)
        ones = ops.constant(1.0, dtype=x.dtype, device=x.device)
        ones = ops.broadcast_to(ones, [self.dim])
        zeros = ops.constant(0.0, dtype=x.dtype, device=x.device)
        zeros = ops.broadcast_to(zeros, [self.dim])
        return ops.layer_norm(x, ones, zeros, epsilon=self.eps)
