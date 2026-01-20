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

"""Layer normalization for module_v3."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

from ..module import Module


class LayerNorm(Module[[Tensor], Tensor]):
    """Layer normalization over the last dimension."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        keep_dtype: bool = True,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ):
        """Initialize LayerNorm.

        Args:
            dim: Size of the last dimension to normalize.
            eps: Numerical stability constant.
            keep_dtype: Whether to preserve input dtype in computation.
            elementwise_affine: Whether to apply learned scale and bias.
            use_bias: Whether to learn an additive bias term.
        """
        super().__init__()
        self.eps = eps
        self.keep_dtype = keep_dtype
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Tensor.ones([dim])
            self.bias = Tensor.zeros([dim]) if use_bias else None
        else:
            self.weight = None
            self.bias = None

    def _affine_params(self, input: Tensor) -> tuple[Tensor, Tensor]:
        if self.weight is None:
            gamma = F.broadcast_to(
                F.constant(1.0, dtype=input.dtype, device=input.device),
                shape=(input.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = F.broadcast_to(
                F.constant(0.0, dtype=input.dtype, device=input.device),
                shape=(input.shape[-1],),
            )
        else:
            beta = self.bias

        return gamma, beta

    def forward(self, input: Tensor) -> Tensor:
        gamma, beta = self._affine_params(input)
        if self.keep_dtype:
            return F.layer_norm(input, gamma=gamma, beta=beta, epsilon=self.eps)
        output = F.layer_norm(
            F.cast(input, DType.float32),
            gamma=F.cast(gamma, DType.float32),
            beta=F.cast(beta, DType.float32),
            epsilon=self.eps,
        )
        return F.cast(output, input.dtype)
