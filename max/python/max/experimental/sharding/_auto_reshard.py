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

"""Decides what an op does when its chosen plan has to redistribute an input.

Every op with a sharding rule offers a set of rows, each naming the
placement every input needs on one mesh axis and the output placement that
follows. :func:`pick_reshard_action` takes one row per mesh axis and
redistributes any input not already placed as that row needs.
:func:`auto_reshard` sets what happens then for the ops inside its block:
``"silent"`` redistributes, ``"warn"`` redistributes and reports each move,
and ``"raise"`` refuses with a :class:`ShardingError` that shows each
axis's candidates and the ``transfer_to`` calls that would make the moves
explicit.
"""

from __future__ import annotations

import contextvars
import sys
import warnings
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Literal

from .action import Action, AxisAssignment
from .cost import feasible_rows_at_axis, tensor_byte_count, transition_cost
from .mappings import DeviceMapping
from .mesh import DeviceMesh
from .placements import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    Transition,
)
from .types import TensorLayout

_AUTO_RESHARD_POLICY: contextvars.ContextVar[
    tuple[frozenset[Transition], str]
] = contextvars.ContextVar(
    "max_auto_reshard_policy", default=(DEFAULT_TRANSITIONS, "raise")
)
"""The allowed transitions and the reshard mode of the innermost block."""

_RESHARD_MODES = ("silent", "warn", "raise")

MeshAxisChoice = tuple[
    tuple[Placement, ...],
    list[tuple[float, AxisAssignment]],
    AxisAssignment | None,
    AxisAssignment,
]
"""What the picker saw and did on one mesh axis.

In order: each input's placement on the axis before the choice, every
feasible row with its cost (cheapest first, then fewest inputs moved, then
the rule's order), the cheapest allowed row (``None`` if no row is
allowed), and the row committed for the axes after it (the allowed row, or
the cheapest row within :data:`DEFAULT_TRANSITIONS` when none is allowed).
"""


@contextmanager
def auto_reshard(
    allowed_transitions: Iterable[Transition] | None = None,
    *,
    mode: Literal["silent", "warn", "raise"] | None = None,
) -> Generator[None]:
    """Sets how ops in this block may change the placement of their inputs.

    On each mesh axis the picker takes the cheapest row whose transitions
    are all in ``allowed_transitions``, and ``mode`` decides what happens
    when that row moves an input. Outside any block the picker allows
    :data:`DEFAULT_TRANSITIONS` and moves inputs silently; an argument
    left out keeps the enclosing block's setting. Use ``"warn"`` to find
    the moves an op makes, and ``"raise"`` to require each one to be
    written as an explicit ``transfer_to``. Also works as a decorator. For
    example:

    .. invisible-code-block: python

        from max.experimental.sharding import (
            ALL_TRANSITIONS,
            Partial,
            Replicated,
            auto_reshard,
        )

    .. code-block:: python

        with auto_reshard(mode="warn"):    # reshard, and say so each time
            ...
        with auto_reshard(mode="raise"):   # require explicit transfer_to
            ...
        with auto_reshard(ALL_TRANSITIONS):      # sequence parallelism too
            ...
        # Only allreduce partial sums.
        with auto_reshard({(Partial, Replicated)}):
            ...

    Args:
        allowed_transitions: The ``(from, to)`` placement-type pairs the
            picker may plan with. A row needing any other transition is
            not allowed; if no row is allowed on some mesh axis, the op
            raises.
        mode: What happens when a chosen row moves an input.
            ``"silent"`` moves it, ``"warn"`` moves it and emits a
            :class:`UserWarning` naming the move and the ``transfer_to``
            that would make it explicit, ``"raise"`` refuses with a
            :class:`ShardingError` saying the same.

    Raises:
        TypeError: If an allowed transition is not a pair of placement
            types.
        ValueError: If ``mode`` is not one of the three above.
    """
    enclosing_allowed, enclosing_mode = _AUTO_RESHARD_POLICY.get()
    allowed = (
        enclosing_allowed
        if allowed_transitions is None
        else frozenset(allowed_transitions)
    )
    for transition in allowed:
        if not (
            isinstance(transition, tuple)
            and len(transition) == 2
            and all(
                isinstance(t, type) and issubclass(t, Placement)
                for t in transition
            )
        ):
            raise TypeError(
                "auto_reshard takes (from, to) pairs of placement types, "
                f"got {transition!r}."
            )
    if mode is not None and mode not in _RESHARD_MODES:
        raise ValueError(
            f"auto_reshard mode must be one of {_RESHARD_MODES}, got {mode!r}."
        )
    token = _AUTO_RESHARD_POLICY.set(
        (allowed, enclosing_mode if mode is None else mode)
    )
    try:
        yield
    finally:
        _AUTO_RESHARD_POLICY.reset(token)


def moved_input_slots(
    current: Sequence[Placement], row: AxisAssignment
) -> list[int]:
    """Returns the inputs whose placement ``row`` changes on one mesh axis."""
    return [
        slot
        for slot, (src, dst) in enumerate(
            zip(current, row.needed_inputs, strict=True)
        )
        if src != dst
    ]


def row_transitions(
    current: Sequence[Placement], row: AxisAssignment
) -> set[Transition]:
    """Returns the ``(from, to)`` placement types ``row`` needs on one axis."""
    return {
        (type(current[slot]), type(row.needed_inputs[slot]))
        for slot in moved_input_slots(current, row)
    }


def pick_reshard_action(
    rows: Sequence[AxisAssignment],
    layouts: tuple[TensorLayout, ...],
    *,
    op_name: str,
    operand_names: Sequence[str],
) -> Action:
    """Picks one row per mesh axis for this op under the policy in scope.

    Decides the mesh axes in order: each axis ranks the rows against the
    placements the axes before it committed, and costs them with its own
    mesh-axis size.

    Args:
        rows: The rows the op's rule offers.
        layouts: The op's tensor operands, in the order the rows place them.
        op_name: The name of the op, used in the report.
        operand_names: The names of the op's tensor operands, used in the
            report.

    Returns:
        The chosen action: the placement every operand takes and the
        placement each result takes.

    Raises:
        ShardingError: If no rows can place this call. This includes when
            the rule returns no rows, a row does not place every operand or
            names a different number of results, the operands sit on meshes
            of different shapes or axis names, no row is feasible or allowed
            on some mesh axis, or the chosen rows move an operand while
            ``mode`` is ``"raise"``.
    """
    if not rows:
        raise ShardingError("A rule must declare at least one placement row.")
    for row in rows:
        if len(row.needed_inputs) != len(layouts):
            raise ShardingError(
                "Every rule row must place every tensor operand."
            )
        if len(row.outputs) != len(rows[0].outputs):
            raise ShardingError(
                "Every rule row must declare the same tensor outputs."
            )
    mesh = next(
        (layout.mesh for layout in layouts if layout.mesh.num_devices > 1),
        layouts[0].mesh,
    )
    if any(
        layout.mesh.num_devices > 1
        and (layout.mesh.mesh_shape, layout.mesh.axis_names)
        != (mesh.mesh_shape, mesh.axis_names)
        for layout in layouts
    ):
        raise ShardingError(
            "A rule's tensor operands must be on meshes with the same shape "
            "and axis names."
        )
    allowed, mode = _AUTO_RESHARD_POLICY.get()
    placements = [
        [
            p[axis] if axis < len(p) else Replicated()
            for axis in range(mesh.ndim)
        ]
        for p in (l.mapping.placements for l in layouts)
    ]
    per_axis: list[MeshAxisChoice] = []
    for axis in range(mesh.ndim):
        current = tuple(p[axis] for p in placements)
        scored: list[tuple[float, int, AxisAssignment]] = []
        for row in dict.fromkeys(
            feasible_rows_at_axis(
                rows, layouts, mesh, axis, [tuple(p) for p in placements]
            )
        ):
            moved = moved_input_slots(current, row)
            cost = sum(
                transition_cost(
                    current[slot],
                    row.needed_inputs[slot],
                    message_bytes=tensor_byte_count(layouts[slot]),
                    mesh=mesh,
                    axis_index=axis,
                )
                for slot in moved
            )
            if cost != float("inf"):
                scored.append((cost, len(moved), row))
        # A stable sort on (cost, moves) keeps the rule's order for ties.
        ranked = [
            (cost, row)
            for cost, _, row in sorted(scored, key=lambda c: (c[0], c[1]))
        ]
        if not ranked:
            raise ShardingError(
                f"{op_name}: the sharding rule returned no feasible plan on "
                f"mesh axis {mesh.axis_names[axis]!r}."
            )
        chosen = next(
            (r for _, r in ranked if row_transitions(current, r) <= allowed),
            None,
        )
        # Commit a row even when none is allowed, so the report's later axes
        # rank against a state the suggested transfer_to could reach.
        committed = chosen or next(
            (
                r
                for _, r in ranked
                if row_transitions(current, r) <= DEFAULT_TRANSITIONS
            ),
            ranked[0][1],
        )
        per_axis.append((current, ranked, chosen, committed))
        for slot, needed in enumerate(committed.needed_inputs):
            placements[slot][axis] = needed

    undecided = any(chosen is None for _, _, chosen, _ in per_axis)
    moves = any(
        moved_input_slots(current, committed)
        for current, _, _, committed in per_axis
    )
    if undecided or (moves and mode != "silent"):
        report = format_reshard_report(
            op_name=op_name,
            operand_names=operand_names,
            operand_layouts=layouts,
            mesh=mesh,
            allowed_transitions=allowed,
            mode=mode,
            per_axis=per_axis,
        )
        if undecided or mode == "raise":
            raise ShardingError(report)
        warnings.warn(report, UserWarning, stacklevel=4)
    committed_rows = [committed for *_, committed in per_axis]
    return Action(
        inputs=_inputs_with_chosen_mappings(layouts, committed_rows),
        outputs=_outputs_with_chosen_mappings(mesh, committed_rows),
    )


def _inputs_with_chosen_mappings(
    layouts: tuple[TensorLayout, ...], rows: Sequence[AxisAssignment]
) -> tuple[DeviceMapping, ...]:
    """Returns one mapping per tensor operand, as the chosen rows place it."""

    def place(slot: int, layout: TensorLayout) -> DeviceMapping:
        own = layout.mapping.mesh
        if own.num_devices == 1:
            return layout.mapping
        # A row names placements, not devices, so an operand stays on its
        # own mesh. Moving it to another mesh is up to the caller's
        # ``transfer_to``.
        return DeviceMapping(
            own, tuple(row.needed_inputs[slot] for row in rows)
        )

    return tuple(place(i, l) for i, l in enumerate(layouts))


def _outputs_with_chosen_mappings(
    mesh: DeviceMesh, rows: Sequence[AxisAssignment]
) -> tuple[DeviceMapping, ...]:
    """Returns one mapping per result, reading each mesh axis's row for it."""
    return tuple(
        DeviceMapping(mesh, tuple(row.outputs[i] for row in rows))
        for i in range(len(rows[0].outputs))
    )


_FRAMEWORK_MODULE_PREFIXES = (
    "max.experimental.functional.",
    "max.experimental.tensor",
    "max.experimental.realization_context",
    "max.experimental.nn.",
    "max.experimental.sharding.",
    "contextlib",
)


def format_reshard_report(
    *,
    op_name: str,
    operand_names: Sequence[str],
    operand_layouts: Sequence[TensorLayout],
    mesh: DeviceMesh,
    allowed_transitions: frozenset[Transition],
    mode: str,
    per_axis: Sequence[MeshAxisChoice],
) -> str:
    """Lays out every input, then the picker's choice on each mesh axis.

    The report lists each input and the region in scope, then one line per
    candidate row on each mesh axis with its placement change, collective,
    bytes moved and status, then the ``transfer_to`` calls and regions that
    would avoid the automatic move, then the first call site outside the
    framework.

    Args:
        op_name: The name of the op being dispatched.
        operand_names: The names of the op's tensor operands, in call order.
        operand_layouts: The layout of each tensor operand, in the same
            order.
        mesh: The device mesh the op dispatches over.
        allowed_transitions: The ``(from, to)`` placement-type pairs the
            picker may plan with.
        mode: The reshard mode in scope, one of ``"silent"``, ``"warn"`` or
            ``"raise"``.
        per_axis: What the picker saw and did on each mesh axis, in
            mesh-axis order.

    Returns:
        The report, ready to print or to carry in a warning or an error.
    """
    names = [
        operand_names[i] if i < len(operand_names) else f"arg{i}"
        for i in range(len(operand_layouts))
    ]
    undecided_axes = [
        axis
        for axis, (_, _, chosen, _) in enumerate(per_axis)
        if chosen is None
    ]
    moved_slots = {
        slot
        for current, _, _, committed in per_axis
        for slot in moved_input_slots(current, committed)
    }
    call = f"{op_name}({', '.join(names)})"
    if undecided_axes:
        head = (
            f"{call}: no resharding on mesh axis "
            f"{mesh.axis_names[undecided_axes[0]]!r} uses only the "
            "transitions this region allows"
        )
    elif mode == "warn":
        # A warning reports moves that happened, so it must not claim a refusal.
        moved = ", ".join(names[s] for s in sorted(moved_slots))
        head = f"{call} reshards {moved}"
    else:
        head = (
            f"{call} needs a resharding; this region does not reshard "
            "automatically"
        )
    width = max(len(n) for n in (*names, "region"))
    lines = [head]
    lines += [
        f"  {n:<{width}} : {_format_layout(l)}"
        for n, l in zip(names, operand_layouts, strict=True)
    ]
    lines.append(
        f"  {'region':<{width}} : "
        f"auto_reshard({_format_transition_set(allowed_transitions)}, "
        f'mode="{mode}")'
    )
    if per_axis:
        lines += ["", "  decided per mesh axis"]
        lines += _format_candidate_table(
            names, mesh, allowed_transitions, per_axis
        )
        lines.append("")
        for slot, name in enumerate(names):
            if slot in moved_slots:
                needed = [
                    committed.needed_inputs[slot] for *_, committed in per_axis
                ]
                mapping = ", ".join(
                    _placement_constructor_expr(p) for p in needed
                )
                lines.append(
                    f"  {name} = transfer_to({name}, DeviceMapping(mesh, "
                    f"({mapping}{',' if len(needed) == 1 else ''})))"
                )
        if undecided_axes:
            widened = allowed_transitions.union(
                *(
                    row_transitions(current, committed)
                    for current, _, _, committed in per_axis
                )
            )
            lines.append(
                f"  with auto_reshard({_format_transition_set(widened)}, "
                'mode="silent"):'
            )
        else:
            lines.append('  with auto_reshard(mode="silent"):')
            if any(ranked[0][1] != chosen for _, ranked, chosen, _ in per_axis):
                lines.append(
                    '  with auto_reshard(ALL_TRANSITIONS, mode="silent"):'
                    "   # also allows the cheaper rows above"
                )
    lines.append(f"  at {_first_user_call_site()}")
    return "\n" + "\n".join(lines)


def _format_candidate_table(
    names: Sequence[str],
    mesh: DeviceMesh,
    allowed_transitions: frozenset[Transition],
    per_axis: Sequence[MeshAxisChoice],
) -> list[str]:
    """Formats one aligned line per candidate row, then the result line."""
    table = [["axis", *names, "-> out", "collective", "bytes", "status"]]
    for axis, (current, ranked, chosen, _) in enumerate(per_axis):
        axis_cell = f'"{mesh.axis_names[axis]}"'
        for cost, row in ranked:
            moved = moved_input_slots(current, row)
            line = [axis_cell]
            for name, src, dst in zip(
                names, current, row.needed_inputs, strict=True
            ):
                change = (
                    _placement_short_code(src)
                    if src == dst
                    else f"{_placement_short_code(src)}->{_placement_short_code(dst)}"
                )
                line.append(f"{name}{{{change}}}")
            line.append(
                "-> "
                + _format_outputs(_placement_short_code(o) for o in row.outputs)
            )
            collectives = [
                f"{current[s].transition_to(row.needed_inputs[s]).value}"
                f"({names[s]})"
                for s in moved
            ]
            line.append(", ".join(collectives) or "keep")
            line.append(f"{cost:g}".rjust(6))
            if row == chosen:
                resharded = ", ".join(names[s] for s in moved)
                line.append(
                    f"chosen, reshards {resharded}" if resharded else "chosen"
                )
            elif not row_transitions(current, row) <= allowed_transitions:
                line.append("not allowed")
            table.append(line)
            axis_cell = ""
    if all(chosen is not None for _, _, chosen, _ in per_axis):
        committed_rows = [committed for *_, committed in per_axis]
        table.append(
            [
                "result",
                *(
                    f"{name}{_format_placements(mesh, (r.needed_inputs[i] for r in committed_rows))}"
                    for i, name in enumerate(names)
                ),
                "-> "
                + _format_outputs(
                    _format_placements(
                        mesh, (r.outputs[i] for r in committed_rows)
                    )
                    for i in range(len(committed_rows[0].outputs))
                ),
            ]
        )
    widths = [
        max(len(line[i]) for line in table if i < len(line))
        for i in range(max(len(line) for line in table))
    ]
    return [
        "  "
        + "  ".join(
            cell.ljust(w) for cell, w in zip(line, widths, strict=False)
        ).rstrip()
        for line in table
    ]


def _format_layout(layout: TensorLayout) -> str:
    """Formats a layout, like ``float32[4, 8]@mesh(tp=2){"tp":S0}``."""
    dtype = str(layout.dtype).removeprefix("DType.")
    shape = ", ".join(str(d) for d in layout.shape)
    mesh = layout.mapping.mesh
    if mesh.num_devices == 1:
        device = mesh.devices[0]
        return f"{dtype}[{shape}]@{device.label}:{device.id}"
    grid = ", ".join(
        f"{n}={s}"
        for n, s in zip(mesh.axis_names, mesh.mesh_shape, strict=True)
    )
    return (
        f"{dtype}[{shape}]@mesh({grid})"
        f"{_format_placements(mesh, layout.mapping.placements)}"
    )


def _format_outputs(codes: Iterable[str]) -> str:
    """Formats one result bare and several as a parenthesized tuple."""
    rendered = list(codes)
    return rendered[0] if len(rendered) == 1 else f"({', '.join(rendered)})"


def _format_placements(
    mesh: DeviceMesh, placements: Iterable[Placement]
) -> str:
    """Formats placements per mesh axis, like ``{"dp":R, "tp":S0}``."""
    return (
        "{"
        + ", ".join(
            f'"{n}":{_placement_short_code(p)}'
            for n, p in zip(mesh.axis_names, placements, strict=False)
        )
        + "}"
    )


def _placement_short_code(p: Placement) -> str:
    """Spells a placement as ``R``, ``S<axis>`` or ``P`` for the table."""
    if isinstance(p, Replicated):
        return "R"
    if isinstance(p, Sharded):
        return f"S{p.axis}"
    if isinstance(p, Partial):
        return "P"
    return repr(p)


def _placement_constructor_expr(p: Placement) -> str:
    """Spells a placement as it is written in a ``DeviceMapping``."""
    if isinstance(p, Replicated):
        return "R"
    if isinstance(p, Partial) and p == Partial():
        return "P"
    if isinstance(p, Sharded):
        return f"Sharded({p.axis})"
    return repr(p)


def _format_transition_set(transitions: frozenset[Transition]) -> str:
    """Names the two transition constants and spells out any other set."""
    if transitions == DEFAULT_TRANSITIONS:
        return "DEFAULT_TRANSITIONS"
    if transitions == ALL_TRANSITIONS:
        return "ALL_TRANSITIONS"
    return (
        "{"
        + ", ".join(
            sorted(f"({s.__name__}, {d.__name__})" for s, d in transitions)
        )
        + "}"
    )


def _first_user_call_site() -> str:
    """Names the first non-framework frame, ``"<file>:<line> in <func>()"``."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "") or ""
        if not any(module.startswith(p) for p in _FRAMEWORK_MODULE_PREFIXES):
            return (
                f"{frame.f_code.co_filename}:{frame.f_lineno} in "
                f"{frame.f_code.co_name}()"
            )
        frame = frame.f_back
    return "<unknown call site>"
