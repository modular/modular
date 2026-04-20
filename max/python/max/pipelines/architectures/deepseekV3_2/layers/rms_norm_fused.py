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
"""RMSNorm with fused residual addition for DeepseekV3.2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue
from max.nn import Module, RMSNorm


class RMSNormFused(Module):
    """RMSNorm with optional fused residual addition.

    When called without a residual, performs standard RMSNorm with float32
    computation. When called with a residual, fuses the residual addition
    with normalization for better performance.

    Uses composition to avoid signature incompatibility with base RMSNorm.
    Weight is accessible at `rms_norm.weight` in the state dict.

    Args:
        dim: Size of the last dimension of the expected input.
        eps: Value added to denominator for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.rms_norm = RMSNorm(dim=dim, dtype=DType.float32, eps=eps)

    def __call__(
        self, x: TensorValue, residual: TensorValue | None = None
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Apply RMSNorm with optional residual fusion.

        Args:
            x: Input tensor.
            residual: Optional residual tensor to add before normalization.

        Returns:
            If residual is None: normalized tensor.
            If residual is provided: tuple of (normalized tensor, fused tensor).
        """
        dtype = x.dtype
        if residual is None:
            x_float = x.cast(DType.float32)
            y = self.rms_norm(x_float)
            return y.cast(dtype)
        else:
            fused = x.cast(DType.float32) + residual.cast(DType.float32)
            y = self.rms_norm(fused)
            return y.cast(dtype), fused.cast(dtype)
