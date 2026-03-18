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

"""Placement types describing how tensor data is distributed across a device mesh.

Each placement is associated with exactly one mesh axis. For an N-dimensional
device mesh, a distributed tensor has N placements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Replicated:
    """Every device on this mesh axis holds the same copy of the data."""

    def __repr__(self) -> str:
        return "Replicated()"


@dataclass(frozen=True)
class Sharded:
    """Every device on this mesh axis holds a slice along `axis`.

    Args:
        axis: The tensor axis along which data is split.
    """

    axis: int

    def __repr__(self) -> str:
        return f"Sharded(axis={self.axis})"


class ReduceOp(str, Enum):
    """Reduction operations for partial placements."""

    SUM = "sum"


@dataclass(frozen=True)
class Partial:
    """Every device holds a partial result that must be reduced.

    Args:
        reduce_op: The reduction operation to apply. Defaults to sum.
    """

    reduce_op: ReduceOp = ReduceOp.SUM

    def __repr__(self) -> str:
        return f"Partial(reduce_op={self.reduce_op.value!r})"


Placement = Replicated | Sharded | Partial
