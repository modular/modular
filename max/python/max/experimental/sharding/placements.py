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

"""Placement types describing how tensor data is distributed across a mesh axis.

Each placement is associated with exactly one mesh axis. For an N-dimensional
device mesh, a distributed tensor has N placements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Placement(ABC):
    """Abstract base for all placement types.

    Every placement describes what a single mesh axis does to a tensor:

    * :class:`Replicated` — full copy on every device.
    * :class:`Sharded` — split along a tensor dimension.
    * :class:`Partial` — partial result needing reduction.

    Custom subclasses (e.g. ``StridedShard``, ``InterleavedShard``) can
    extend the vocabulary for non-standard partitioning patterns.
    """

    @abstractmethod
    def __repr__(self) -> str: ...


class Replicated(Placement):
    """Every device on this mesh axis holds the same copy of the data.

    Singleton — ``Replicated() is Replicated()`` is always ``True``.
    """

    _instance: Replicated | None = None

    def __new__(cls) -> Replicated:
        """Returns the singleton Replicated instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Replicated()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replicated)

    def __hash__(self) -> int:
        return hash("Replicated")


@dataclass(frozen=True)
class Sharded(Placement):
    """Every device on this mesh axis holds a slice along ``axis``.

    Args:
        axis: The tensor axis along which data is split.
    """

    axis: int

    def __repr__(self) -> str:
        return f"Sharded(axis={self.axis})"


class ReduceOp(str, Enum):
    """Reduction operations for partial placements.

    Matches the standard set from PyTorch's ``c10d::ReduceOp``.

    Note: Only ``SUM`` is currently used in eager dispatch.  The others
    are defined for forward-compatibility but do not yet have
    corresponding MAX reduce op implementations.
    """

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class Partial(Placement):
    """Every device holds a partial result that must be reduced.

    Cached per ``reduce_op`` — ``Partial() is Partial()`` is ``True``.

    Args:
        reduce_op: The reduction operation to apply. Defaults to sum.
    """

    _cache: dict[ReduceOp, Partial] = {}
    _reduce_op: ReduceOp

    def __new__(cls, reduce_op: ReduceOp = ReduceOp.SUM) -> Partial:
        """Returns a cached Partial instance for the given reduce_op."""
        if reduce_op not in cls._cache:
            instance = super().__new__(cls)
            instance._reduce_op = reduce_op
            cls._cache[reduce_op] = instance
        return cls._cache[reduce_op]

    @property
    def reduce_op(self) -> ReduceOp:
        """The reduction operation applied when combining partial results."""
        return self._reduce_op

    def __repr__(self) -> str:
        return f"Partial(reduce_op={self.reduce_op.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Partial) and self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return hash(("Partial", self.reduce_op))
