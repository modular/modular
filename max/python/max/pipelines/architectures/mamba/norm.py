# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Non-legacy RMSNorm modules for Mamba architecture."""

from __future__ import annotations

from max.graph import Dim
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

from .functional_ops import rms_norm_fused_residual as _rms_norm_fused_residual


class RMSNormFusedResidual(Module[[Tensor, Tensor], tuple[Tensor, Tensor]]):
    """Fused RMSNorm with residual addition: norm(x + residual).

    Combines the residual addition and RMSNorm into a single fused kernel
    call for better performance. Used for all blocks after the first one
    in the Mamba architecture.
    """

    weight: Tensor
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    @property
    def dim(self) -> Dim:
        return self.weight.shape[0]

    def forward(
        self, x: Tensor, residual: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply fused residual addition and RMSNorm.

        Args:
            x: Input tensor of shape (*, hidden).
            residual: Residual tensor of shape (*, hidden).

        Returns:
            Tuple of (normalized, updated_residual) where:
                normalized: RMSNorm(x + residual) of shape (*, hidden)
                updated_residual: x + residual of shape (*, hidden)
        """
        return _rms_norm_fused_residual(
            x, residual, self.weight, self.eps
        )


class RMSNormForFirstBlock(Module[[Tensor], Tensor]):
    """Plain RMSNorm for the first Mamba block (no incoming residual).

    The first block has no previous residual to fuse, so it uses standard
    RMSNorm. This reuses the non-legacy rms_norm function from max.nn.
    """

    weight: Tensor
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    @property
    def dim(self) -> Dim:
        return self.weight.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)
