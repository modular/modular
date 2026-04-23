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

"""Row- and column-parallel linear layers with pre-defined sharding intent.

These layers extend :class:`~max.experimental.nn.linear.Linear` by tagging
the weight tensor with device placements, which come into use when the Module
is transferred to a multi-device mesh.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn.linear import Linear
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import DimLike

from .mesh_axis import TP


class _DistributedLinear(Linear):
    """Temporary class that redefines Linear.forward() to use F.

    Will be removed when F is merged in with F.
    """

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        """Applies row-parallel linear transformation.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.

        Returns:
            Output tensor with shape ``(..., out_dim)``.
        """
        y = F.matmul(x, F.transpose(self.weight, -1, -2))
        if isinstance(self.bias, Tensor):
            y = F.add(y, self.bias)
        return y


class RowParallelLinear(_DistributedLinear):
    """Linear layer with row-parallel weight sharding."""

    def __init__(
        self,
        in_dim: DimLike,
        out_dim: DimLike,
        *,
        bias: bool = True,
    ):
        super().__init__(in_dim, out_dim, bias=bias)
        self.weight._mapping = NamedMapping(self.weight.mesh, (None, TP))

        if isinstance(self.bias, Tensor):
            self.bias._mapping = NamedMapping(self.bias.mesh, (None,))


class ColumnParallelLinear(_DistributedLinear):
    """Linear layer with column-parallel weight sharding."""

    def __init__(
        self,
        in_dim: DimLike,
        out_dim: DimLike,
        *,
        bias: bool = True,
    ):
        super().__init__(in_dim, out_dim, bias=bias)
        self.weight._mapping = NamedMapping(self.weight.mesh, (TP, None))

        if isinstance(self.bias, Tensor):
            self.bias._mapping = NamedMapping(self.bias.mesh, (TP,))
