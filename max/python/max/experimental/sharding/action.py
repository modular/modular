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

"""Per-op decision data model: :class:`AxisAssignment`, :class:`Action`, :class:`ActionSet`."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

from max.experimental.sharding.mappings import DeviceMapping
from max.experimental.sharding.mesh import DeviceMesh
from max.experimental.sharding.placements import Placement
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim


class AxisAssignment(NamedTuple):
    """One per-mesh-axis row in an :class:`ActionSet`.

    Reads as: given these per-axis input placements, the output along
    that axis is :attr:`output`. Picking one :class:`AxisAssignment`
    per mesh axis builds a multi-axis :class:`Action`.
    """

    needed_inputs: tuple[Placement, ...]
    output: Placement


_Value = TypeVar("_Value")


class PerShard(Generic[_Value]):
    """One value per device of a mesh, in the mesh's row-major device order.

    :func:`~max.experimental.functional.call_on_mesh` passes each device its
    own entry, for example a device's local shape, its vocabulary bounds or
    whether it is the first device. It is its own type because a plain
    tuple argument reaches every device whole. A rule also returns one in
    :attr:`ActionSet.extras` for an argument that differs per device; any
    other value there is the same on every device.
    """

    __slots__ = ("values",)

    values: tuple[_Value, ...]

    def __init__(self, values: Iterable[_Value]) -> None:
        object.__setattr__(self, "values", tuple(values))

    def __getitem__(self, i: int) -> _Value:
        return self.values[i]

    def __iter__(self) -> Iterator[_Value]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"PerShard({self.values!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PerShard) and self.values == other.values

    def __hash__(self) -> int:
        return hash(("PerShard", self.values))


@dataclass(frozen=True)
class Action:
    """A rule's picked decision for one op call.

    :attr:`inputs` has one entry per op argument in positional order:
    :class:`DeviceMapping` at tensor positions; bare scalar (treated as
    uniform across ranks) or :class:`PerShard` at non-tensor positions.
    :attr:`outputs` has one :class:`DeviceMapping` per op output. The
    dispatcher inserts ``transfer_to`` collectives to match
    :attr:`inputs` before per-shard dispatch; :attr:`outputs` is the
    post-op mapping each result wears.
    """

    inputs: tuple[Any, ...]
    outputs: tuple[DeviceMapping, ...]

    def __iter__(self) -> Iterator[Any]:
        """Yields ``(inputs, outputs)``."""
        yield self.inputs
        yield self.outputs


@dataclass(frozen=True)
class ActionSet:
    """A rule's per-axis sharding options for one op call.

    Shape-aware (it depends on operand layouts) but cost-blind: it
    lists what is possible, not what is cheapest. The dispatcher
    picks one entry per mesh axis.
    """

    axis_assignments: tuple[AxisAssignment, ...]
    """Per-axis rows, each pickable independently per mesh axis. The
    last entry is the universal ``(R,…,R) -> R`` fallback."""

    layouts: tuple[TensorLayout, ...]
    """Per-tensor input layouts this menu was built for."""

    mesh: DeviceMesh
    """The mesh both the picker and planner work over."""

    extras: tuple[Any, ...] = ()
    """Non-tensor positional args appended after tensor-input mappings
    in the picked :class:`Action`'s :attr:`inputs`. A bare value is
    treated as uniform across ranks; wrap in :class:`PerShard` to vary
    per rank."""

    result_shape: Sequence[Dim] | None = None
    """Output shape constraint, set by reshape-style rules. Lets the
    feasibility check reject output :class:`Sharded` rows whose result
    dim is too small."""

    finalize: Callable[[Action], Action] | None = None
    """Optional post-pick transform applied as ``finalize(action)``. Used
    by rules that need the picked placement to compute per-rank metadata
    or repack inputs into a user-facing container. Rules pre-bind any
    per-op context by closing over it, so no separate context field is
    needed."""
