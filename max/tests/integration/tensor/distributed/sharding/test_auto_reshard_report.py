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
"""Tests for the report ``auto_reshard`` raises or warns with."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    TensorLayout,
    Transition,
    auto_reshard,
)
from max.experimental.sharding._auto_reshard import (
    _format_transition_set,
    pick_reshard_action,
)

R, P, S0, S1 = Replicated(), Partial(), Sharded(0), Sharded(1)


def mesh(*shape: int) -> DeviceMesh:
    names = ("dp", "tp")[-len(shape) :]
    return DeviceMesh(tuple(CPU() for _ in range(sum(shape))), shape, names)


def layout(
    mesh: DeviceMesh, shape: tuple[int, ...], *placements: Placement
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placements))


def message(
    *layouts: TensorLayout,
    rows: Sequence[AxisAssignment],
    names: Sequence[str] = ("x",),
    allowed: frozenset[Transition] = DEFAULT_TRANSITIONS,
    mode: Literal["warn", "raise"] = "raise",
) -> str:
    """Returns the report the picker raises or warns with for ``rows``."""
    with auto_reshard(allowed, mode=mode):
        if mode == "warn":
            with pytest.warns(UserWarning) as record:
                pick_reshard_action(
                    rows, layouts, op_name="custom_op", operand_names=names
                )
            return str(record[0].message)
        with pytest.raises(ShardingError) as info:
            pick_reshard_action(
                rows, layouts, op_name="custom_op", operand_names=names
            )
        return str(info.value)


def line_with(text: str, needle: str) -> str:
    """Returns the one line containing ``needle``, its padding collapsed."""
    (line,) = [l for l in text.splitlines() if needle in l]
    return " ".join(line.split())


PARTIAL_ROWS = [AxisAssignment((R,), (R,)), AxisAssignment((S0,), (S0,))]


def test_names_the_call_and_lays_out_every_input() -> None:
    text = message(layout(mesh(4), (1024,), P), rows=PARTIAL_ROWS)
    assert text.startswith(
        "\ncustom_op(x) needs a resharding; this region does not reshard "
        "automatically"
    )
    assert '  x      : float32[1024]@mesh(tp=4){"tp":P}' in text
    assert '  region : auto_reshard(DEFAULT_TRANSITIONS, mode="raise")' in text
    assert "  at " in text


def test_a_warning_says_what_it_resharded() -> None:
    """A warning reports a move that happened, not a refusal."""
    text = message(layout(mesh(4), (1024,), P), rows=PARTIAL_ROWS, mode="warn")
    assert text.startswith("\ncustom_op(x) reshards x")
    assert "does not reshard automatically" not in text


def test_ranks_candidates_cheapest_first_and_labels_each() -> None:
    text = message(layout(mesh(4), (1024,), P), rows=PARTIAL_ROWS)
    cheaper, chosen = [
        " ".join(l.split()) for l in text.splitlines() if "x{P->" in l
    ]
    assert cheaper.startswith('"tp" x{P->S0} -> S0 reduce_scatter(x)')
    assert cheaper.endswith("not allowed")
    assert "x{P->R} -> R allreduce(x)" in chosen
    assert chosen.endswith("chosen, reshards x")
    assert 'x{"tp":R} -> {"tp":R}' in line_with(text, "result")


def test_an_unchanged_input_keeps_its_code_in_the_cell() -> None:
    text = message(
        layout(mesh(2), (4, 8), S1),
        layout(mesh(2), (8, 16), R),
        rows=[
            AxisAssignment((R, R), (R,)),
            AxisAssignment((S0, R), (S0,)),
            AxisAssignment((R, S1), (S1,)),
            AxisAssignment((S1, S0), (P,)),
        ],
        names=("lhs", "rhs"),
    )
    chosen = line_with(text, "chosen")
    assert "lhs{S1} rhs{R->S0} -> P local_slice(rhs)" in chosen
    assert chosen.endswith("chosen, reshards rhs")
    assert (
        "  rhs = transfer_to(rhs, DeviceMapping(mesh, (Sharded(0),)))" in text
    )
    assert "lhs = transfer_to" not in text


def test_a_2d_mesh_gets_one_block_per_axis() -> None:
    text = message(
        layout(mesh(2, 2), (16, 16), P, P),
        rows=[AxisAssignment((R,), (R,)), AxisAssignment((S0,), (S0,))],
    )
    lines = text.splitlines()
    dp = next(i for i, l in enumerate(lines) if l.startswith('  "dp"'))
    tp = next(i for i, l in enumerate(lines) if l.startswith('  "tp"'))
    assert dp < tp
    assert text.count("chosen, reshards x") == 2
    assert 'x{"dp":R, "tp":R} -> {"dp":R, "tp":R}' in line_with(text, "result")
    assert "  x = transfer_to(x, DeviceMapping(mesh, (R, R)))" in text


def test_tells_the_user_both_ways_out() -> None:
    text = message(layout(mesh(4), (1024,), P), rows=PARTIAL_ROWS)
    assert "  x = transfer_to(x, DeviceMapping(mesh, (R,)))" in text
    assert '  with auto_reshard(mode="silent"):' in text
    assert '  with auto_reshard(ALL_TRANSITIONS, mode="silent"):' in text, (
        "a cheaper row was not allowed"
    )


def test_no_cheaper_row_means_no_widening_hint() -> None:
    text = message(
        layout(mesh(4), (1024,), P), rows=[AxisAssignment((R,), (R,))]
    )
    assert "ALL_TRANSITIONS" not in text


def test_a_narrowed_region_is_told_what_to_add() -> None:
    text = message(
        layout(mesh(4), (1024,), P),
        rows=[AxisAssignment((R,), (R,))],
        allowed=frozenset({(Sharded, Replicated)}),
    )
    assert (
        "custom_op(x): no resharding on mesh axis 'tp' uses only the "
        "transitions this region allows" in text
    )
    assert (
        '  region : auto_reshard({(Sharded, Replicated)}, mode="raise")' in text
    )
    assert "chosen" not in text and "result" not in text
    assert (
        "  with auto_reshard({(Partial, Replicated), (Sharded, Replicated)}, "
        'mode="silent"):' in text
    )


def test_a_single_device_input_shows_its_own_mesh() -> None:
    """Its placement is its own mapping's, and the pick never moves it."""
    text = message(
        layout(mesh(2), (16,), P),
        layout(DeviceMesh.single(CPU()), (16,), R),
        rows=[AxisAssignment((R, R), (R,))],
        names=("x", "w"),
    )
    assert "  w      : float32[16]@cpu:0" in text
    assert "x{P->R} w{R} -> R allreduce(x)" in line_with(text, "chosen")
    assert "w = transfer_to" not in text


def test_an_axis_with_no_allowed_row_still_constrains_the_next() -> None:
    """The suggested transfer_to must be reachable: extent 2 fits one axis."""
    text = message(
        layout(mesh(2, 2), (2, 16), R, R),
        rows=[AxisAssignment((S0,), (S0,)), AxisAssignment((S1,), (S1,))],
        allowed=frozenset({(Partial, Replicated)}),
    )
    assert (
        "  x = transfer_to(x, DeviceMapping(mesh, (Sharded(0), Sharded(1))))"
        in text
    )


def test_transition_sets_name_the_two_constants_and_spell_out_the_rest() -> (
    None
):
    assert _format_transition_set(ALL_TRANSITIONS) == "ALL_TRANSITIONS"
    assert _format_transition_set(DEFAULT_TRANSITIONS) == "DEFAULT_TRANSITIONS"
    assert _format_transition_set(
        frozenset({(Partial, Replicated), (Sharded, Sharded)})
    ) == ("{(Partial, Replicated), (Sharded, Sharded)}")


def test_the_table_names_its_columns() -> None:
    """A reader should not have to guess what the numbers are."""
    text = message(layout(mesh(4), (1024,), P), rows=PARTIAL_ROWS)
    assert line_with(text, "-> out") == "axis x -> out collective bytes status"
