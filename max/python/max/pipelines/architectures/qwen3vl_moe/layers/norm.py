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
"""The decoder's RMS norm."""

from __future__ import annotations

from max.dtype import DType
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

__all__ = ["RMSNorm"]


class RMSNorm(Module[[Tensor], Tensor]):
    """RMS normalization over the last axis, scaled by ``weight``.

    The weight is cast to the input's dtype. The block norms cast the
    normalized value back before the scale; the Q and K norms scale first.
    """

    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        dtype: DType,
        multiply_before_cast: bool = False,
    ) -> None:
        self.weight = Tensor.zeros([dim], dtype=dtype)
        self.eps = eps
        self.multiply_before_cast = multiply_before_cast

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized and scaled, in ``x``'s dtype."""
        return rms_norm(
            x,
            self.weight.cast(x.dtype),
            self.eps,
            multiply_before_cast=self.multiply_before_cast,
        )
