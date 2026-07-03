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

"""Row- and column-parallel linear layers with pre-defined sharding intent."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from max.experimental.nn.linear import Linear
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor

from .mesh_axis import TP


class _LinearProtocol(Protocol):
    weight: Tensor
    bias: Tensor | Literal[0]


_LinearLayer = TypeVar("_LinearLayer", bound=_LinearProtocol)


def col_parallel(layer: _LinearLayer) -> _LinearLayer:
    """Parallelize the linear layer across the column dimension."""
    # Note that the first dimension of the weight is sharded because MAX's
    # linear layer applies ``x @ W.T`` (the transpose of W)
    layer.weight._mapping = NamedMapping(layer.weight.mesh, (TP, None))
    if isinstance(layer.bias, Tensor):
        layer.bias._mapping = NamedMapping(layer.bias.mesh, (TP,))
    return layer


def row_parallel(layer: _LinearLayer) -> _LinearLayer:
    """Parallelize the linear layer across the row dimension."""
    layer.weight._mapping = NamedMapping(layer.weight.mesh, (None, TP))
    if isinstance(layer.bias, Tensor):
        layer.bias._mapping = NamedMapping(layer.bias.mesh, (None,))
    return layer


class ColumnParallelLinear(Linear):
    """Linear layer with column-parallel weight sharding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        col_parallel(self)


class RowParallelLinear(Linear):
    """Linear layer with row-parallel weight sharding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        row_parallel(self)
