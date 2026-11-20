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
"""Qwen3.5's RMSNorm, which keeps its weight at the checkpoint's dtype."""

from __future__ import annotations

from max.dtype import DType
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

__all__ = ["RMSNorm", "rms_norm_scaled"]


def rms_norm_scaled(
    x: Tensor, weight: Tensor, *, eps: float, weight_offset: float = 1.0
) -> Tensor:
    """Returns ``x`` RMS-normalized, cast back, then scaled by ``weight``.

    The scale is ``weight_offset + weight``, applied in ``x``'s dtype.
    """
    return rms_norm(
        x,
        weight.cast(x.dtype),
        eps,
        weight_offset=weight_offset,
        multiply_before_cast=False,
    )


class RMSNorm(Module[[Tensor], Tensor]):
    """RMS normalization over the last axis, cast back before the scale.

    Every Qwen3.5 norm scales by ``1 + weight``, except the gated DeltaNet's
    output norm, which scales by ``weight`` (``weight_offset=0``).
    """

    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        dtype: DType,
        weight_offset: float = 1.0,
    ) -> None:
        self.weight = Tensor.zeros([dim], dtype=dtype)
        self.eps = eps
        self.weight_offset = weight_offset

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized and scaled, in ``x``'s dtype."""
        return rms_norm_scaled(
            x, self.weight, eps=self.eps, weight_offset=self.weight_offset
        )
