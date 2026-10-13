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

"""Defines the rows a sharding rule returns and the functions that build them.

A rule returns :class:`AxisAssignment` rows, and the op picks one row per
mesh axis to form an :class:`Action`. :class:`PerShard` holds one argument
value per device for :func:`~max.experimental.functional.call_on_mesh`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

from max import tree
from max.experimental.sharding.mappings import DeviceMapping
from max.experimental.sharding.placements import (
    Placement,
    Replicated,
    ShardingError,
)
from max.experimental.sharding.types import TensorLayout


class AxisAssignment(NamedTuple):
    """One placement choice that a sharding rule offers for its op.

    A sharding rule is a function with the same parameters as its op. It
    receives the op's arguments with each tensor, at any depth, replaced by
    its :class:`TensorLayout`, and returns a list of rows. Each row gives a
    placement for every tensor operand, in argument order, and for every
    result. Some rows of the matmul rule:

    .. code-block:: python

        from max.experimental.sharding import (
            AxisAssignment,
            Partial,
            Replicated,
            Sharded,
            TensorLayout,
        )

        def matmul_rule(
            lhs: TensorLayout, rhs: TensorLayout
        ) -> list[AxisAssignment]:
            return [
                AxisAssignment((Replicated(), Replicated()), (Replicated(),)),
                # Column parallel: each device holds some of the columns.
                AxisAssignment((Replicated(), Sharded(1)), (Sharded(1),)),
                # Row parallel: each device holds part of the sum.
                AxisAssignment((Sharded(1), Sharded(0)), (Partial(),)),
            ]

    On each mesh axis, the op picks the row whose input moves cost least,
    moves its operands there, and gives its results the row's placements.
    A rule never changes the op's other arguments.
    """

    needed_inputs: tuple[Placement, ...]
    """The placement of each tensor operand, in argument order."""

    outputs: tuple[Placement, ...]
    """The placement of each result, in result order."""


_Value = TypeVar("_Value")


class PerShard(Generic[_Value]):
    """One value per device of a mesh, in the mesh's row-major device order.

    :func:`~max.experimental.functional.call_on_mesh` passes each device its
    own entry, for example a device's local shape, its vocabulary bounds or
    whether it is the first device. It is its own type because a plain
    tuple argument reaches every device whole.
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
    """The placements picked for one op call, one row per mesh axis.

    The op moves each operand to its mapping in :attr:`inputs` before it
    runs and gives its results the mappings in :attr:`outputs`.
    """

    inputs: tuple[DeviceMapping, ...]
    """One mapping per tensor operand, in argument order."""

    outputs: tuple[DeviceMapping, ...]
    """One mapping per result."""

    def __iter__(self) -> Iterator[Any]:
        """Yields ``(inputs, outputs)``."""
        yield self.inputs
        yield self.outputs


def input_placements_at_axis(
    per_input_placements: Sequence[tuple[Placement, ...]], mesh_axis: int
) -> tuple[Placement, ...]:
    """Returns each input's placement on ``mesh_axis``, or ``Replicated``."""
    return tuple(
        p[mesh_axis] if len(p) > mesh_axis else Replicated()
        for p in per_input_placements
    )


def tensor_layouts_in(inputs: tuple[Any, ...]) -> tuple[TensorLayout, ...]:
    """Returns the tensor layouts in ``inputs``, at any depth, in rule order.

    Args:
        inputs: The op's arguments, which may nest tensors in containers.

    Returns:
        One layout per tensor, in the order a walk of ``inputs`` reaches
        them, which is the order a rule's rows name them.
    """
    return tuple(tree.leaves(inputs, leaf=TensorLayout))


def pass_through_rows(
    layouts: Sequence[TensorLayout],
    outputs: Callable[[tuple[Placement, ...]], tuple[Placement, ...] | None],
) -> list[AxisAssignment]:
    """Returns the rows of an op that never moves its inputs.

    Each mesh axis gets the row that keeps every input where it is, with
    the result placements ``outputs`` gives for those input placements.

    Args:
        layouts: The op's tensor operands, in call order.
        outputs: A callable mapping one mesh axis's input placements to
            each result's placement there, or to :obj:`None` where the op
            cannot run on them.

    Returns:
        One row per distinct mesh-axis placement, in mesh-axis order.

    Raises:
        ShardingError: If ``outputs`` returns :obj:`None` on some mesh axis.
    """
    placements = [layout.placements for layout in layouts]
    rows: list[AxisAssignment] = []
    for mesh_axis in range(max(len(p) for p in placements)):
        actuals = input_placements_at_axis(placements, mesh_axis)
        if (out := outputs(actuals)) is None:
            raise ShardingError(
                f"This op cannot run on inputs placed {actuals} on mesh axis "
                f"{mesh_axis}, and it never moves its inputs; place them "
                "with transfer_to first."
            )
        row = AxisAssignment(actuals, out)
        if row not in rows:
            rows.append(row)
    return rows


def replicated_rows(*args: Any, output_count: int = 1) -> list[AxisAssignment]:
    """Returns the one row of an op that runs only on whole tensors.

    Args:
        *args: The op's arguments, as the rule receives them.
        output_count: The number of results. Defaults to ``1``.

    Returns:
        A single row placing every operand and result :class:`Replicated`.

    Raises:
        ValueError: If ``args`` holds no tensor operand.
    """
    if not (layouts := tensor_layouts_in(args)):
        raise ValueError("replicated_rows: at least one layout required.")
    return [
        AxisAssignment(
            (Replicated(),) * len(layouts), (Replicated(),) * output_count
        )
    ]


def match_operand_placement(
    graph_op: Callable[..., Any], name: str, *, output_count: int = 1
) -> Callable[..., list[AxisAssignment]]:
    """Returns a sharding rule whose results take operand ``name``'s placement.

    The rule never moves an operand. It suits an op whose results are laid
    out like one of its inputs, such as per-row values computed from a
    row-sharded tensor.

    Args:
        graph_op: The op the rule is for, read for its parameter names.
        name: The parameter whose placement the results take.
        output_count: The number of results. Defaults to ``1``.

    Returns:
        A rule that keeps every operand where it is and places each result
        as ``name`` is placed.

    Raises:
        ValueError: If ``graph_op`` has no parameter ``name``.
    """
    signature = inspect.signature(graph_op)
    op_name = getattr(graph_op, "__name__", repr(graph_op))
    if name not in signature.parameters:
        raise ValueError(
            f"{op_name!r} has no parameter {name!r} to take a placement "
            f"from; it has {tuple(signature.parameters)}."
        )

    def rule(*args: Any, **kwargs: Any) -> list[AxisAssignment]:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        layouts = tensor_layouts_in(tuple(bound.arguments.values()))
        claimed = bound.arguments[name]
        which = next(
            (i for i, layout in enumerate(layouts) if layout is claimed),
            None,
        )
        if which is None:
            raise ShardingError(
                f"{name!r} holds {claimed!r}, so it has no placement for the "
                f"results of {op_name!r} to take; name a tensor parameter "
                "instead."
            )
        return pass_through_rows(
            layouts, lambda actuals: (actuals[which],) * output_count
        )

    rule.__name__ = f"{op_name}_rule"
    return rule
