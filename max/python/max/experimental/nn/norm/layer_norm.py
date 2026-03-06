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

"""Layer normalization."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

from ..module import Module


def layer_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float,
    keep_dtype: bool,
) -> Tensor:
    """Applies Layer Normalization to an input tensor.

    Args:
        x: Input tensor to normalize.
        gamma: Scale tensor for elementwise affine transform.
        beta: Bias tensor for elementwise affine transform.
        eps: Numerical stability constant.
        keep_dtype: Whether to preserve input dtype in computation.

    Returns:
        A layer-normalized tensor with the same shape and type as `x`.
    """
    if keep_dtype:
        return F.layer_norm(x, gamma=gamma, beta=beta, epsilon=eps)
    output = F.layer_norm(
        F.cast(x, DType.float32),
        gamma=F.cast(gamma, DType.float32),
        beta=F.cast(beta, DType.float32),
        epsilon=eps,
    )
    return F.cast(output, x.dtype)


class LayerNorm(Module[[Tensor], Tensor]):
    """Layer normalization over the last dimension."""

    weight: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        keep_dtype: bool = True,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        """Initialize LayerNorm.

        Args:
            dim: Size of the last dimension to normalize.
            eps: Numerical stability constant.
            keep_dtype: Whether to preserve input dtype in computation.
            elementwise_affine: Whether to apply learned scale.
            use_bias: Whether to apply a bias. It's only effective if elementwise_affine is True.
        """
        super().__init__()
        self.dim = dim
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

    def __rich_repr__(self):
        """Repr matching the Linear constructor."""
        yield "dim", self.dim
        yield "eps", self.eps, 1e-5

    def _affine_params(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.weight is None:
            gamma = F.broadcast_to(
                F.constant(1.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = F.broadcast_to(
                F.constant(0.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            beta = self.bias

        return gamma, beta

    def forward(self, x: Tensor) -> Tensor:
        """Applies layer normalization to the input."""
        gamma, beta = self._affine_params(x)
        return layer_norm(x, gamma, beta, self.eps, self.keep_dtype)
