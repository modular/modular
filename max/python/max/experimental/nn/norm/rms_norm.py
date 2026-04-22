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

"""Root mean square layer normalization."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, ops

from ..module import Module

rms_norm = F.functional(ops.rms_norm)
"""Applies Root Mean Square layer normalization to an input tensor.

See :func:`max.graph.ops.rms_norm` for details.
"""


class RMSNorm(Module[[Tensor], Tensor]):
    """Computes the Root Mean Square normalization on inputs."""

    weight: Tensor
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Constructs RMSNorm.

        Args:
            dim: Size of last dimension of the expected input.
            eps: Value added to denominator for numerical stability.
        """
        self.weight = Tensor.ones([dim])
        self.eps = eps

    @property
    def dim(self) -> Dim:
        """Returns the embedding dimension."""
        return self.weight.shape[0]

    def __rich_repr__(self):
        """Repr matching the Linear constructor."""
        yield "dim", self.dim
        yield "eps", self.eps, 1e-6

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMS normalization to the input."""
        return rms_norm(x, self.weight, self.eps)


class GemmaRMSNorm(RMSNorm):
    """Computes the Root Mean Square normalization on inputs.

    Differences to traditional RMSNorm:
    - x * (1 + w) instead of x * w.
    - (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies Gemma-style RMS normalization to the input."""
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=1.0,
            multiply_before_cast=True,
        )
