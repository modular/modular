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
"""Tests for :mod:`max.experimental.sharding.cost`."""

from __future__ import annotations

import math

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    ShardingError,
    TensorLayout,
    Unknown,
    match_operand_placement,
    replicated_rows,
)
from max.experimental.sharding._auto_reshard import pick_reshard_action
from max.experimental.sharding.action import Action
from max.experimental.sharding.cost import (
    P,
    R,
    feasible_rows_at_axis,
    tensor_byte_count,
    transition_cost,
)
from max.experimental.sharding.placements import Placement


def mesh_1d(n: int) -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), ("tp",))


def layout(
    mesh: DeviceMesh,
    shape: tuple[int, ...],
    placement: tuple[Placement, ...],
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placement))


class TestSingletons:
    def test_r_and_p_are_canonical_instances(self) -> None:
        assert R == Replicated()
        assert P == Partial()


class TestTensorByteCount:
    def test_product_of_shape_times_dtype(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (4, 8, 3), (Replicated(),))
        assert tensor_byte_count(lay) == 4 * 8 * 3 * 4.0

    def test_scalar_is_one_element(self) -> None:
        mesh = mesh_1d(2)
        assert tensor_byte_count(layout(mesh, (), (Replicated(),))) == 4.0


def pick(
    rows: list[AxisAssignment], layouts: tuple[TensorLayout, ...]
) -> Action:
    return pick_reshard_action(
        rows, layouts, op_name="custom_op", operand_names=()
    )


class TestRuleRows:
    """What the picker checks about the rows a rule returns."""

    def test_rejects_no_rows(self) -> None:
        lay = layout(mesh_1d(2), (4,), (Replicated(),))
        with pytest.raises(ShardingError, match="at least one"):
            pick([], (lay,))

    def test_rejects_incomplete_input_rows(self) -> None:
        lay = layout(mesh_1d(2), (4,), (Replicated(),))
        with pytest.raises(ShardingError, match="every tensor operand"):
            pick([AxisAssignment((R,), (R,))], (lay, lay))

    def test_rejects_inconsistent_result_counts(self) -> None:
        lay = layout(mesh_1d(2), (4,), (Replicated(),))
        with pytest.raises(ShardingError, match="same tensor outputs"):
            pick(
                [AxisAssignment((R,), (R,)), AxisAssignment((P,), (P, P))],
                (lay,),
            )


class TestReplicatedRows:
    def test_emits_a_single_all_replicated_row(self) -> None:
        mesh = mesh_1d(2)
        a = layout(mesh, (4,), (Replicated(),))
        b = layout(mesh, (4,), (Replicated(),))
        assert replicated_rows(a, b, 3) == [
            AxisAssignment((Replicated(), Replicated()), (Replicated(),))
        ]

    def test_requires_at_least_one_layout(self) -> None:
        with pytest.raises(ValueError):
            replicated_rows()


class TestMatchOperandPlacement:
    def test_results_take_the_named_operand_placement(self) -> None:
        def op(values: object, offsets: object) -> None: ...

        rule = match_operand_placement(op, "offsets", output_count=2)
        mesh = mesh_1d(2)
        values = layout(mesh, (4, 8), (Replicated(),))
        offsets = layout(mesh, (4,), (Sharded(0),))
        assert rule(values, offsets) == [
            AxisAssignment((Replicated(), Sharded(0)), (Sharded(0),) * 2)
        ]


def _ring_factor(n: int) -> float:
    return 0.0 if n <= 1 else (n - 1) / n


class TestTransitionCost:
    """Ring-collective arithmetic, exercised via :func:`transition_cost`."""

    def test_self_transition_is_free(self) -> None:
        mesh = mesh_1d(4)
        assert (
            transition_cost(
                Sharded(0),
                Sharded(0),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            == 0.0
        )

    def test_replicated_to_sharded_is_local_slice(self) -> None:
        mesh = mesh_1d(4)
        assert (
            0.0
            < transition_cost(
                Replicated(),
                Sharded(0),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            < transition_cost(
                Sharded(0),
                Replicated(),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
        )

    def test_sharded_to_replicated_matches_ring_allgather(self) -> None:
        for n in (2, 4, 8):
            mesh = mesh_1d(n)
            c = transition_cost(
                Sharded(0),
                Replicated(),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            assert math.isclose(c, 1024.0 * _ring_factor(n))

    def test_partial_to_replicated_is_twice_partial_to_sharded(self) -> None:
        mesh = mesh_1d(8)
        ar = transition_cost(
            Partial(),
            Replicated(),
            message_bytes=4096.0,
            mesh=mesh,
            axis_index=0,
        )
        rs = transition_cost(
            Partial(),
            Sharded(0),
            message_bytes=4096.0,
            mesh=mesh,
            axis_index=0,
        )
        assert math.isclose(ar, 2.0 * rs)

    def test_single_device_rings_are_free(self) -> None:
        mesh = mesh_1d(1)
        for src, dst in (
            (Sharded(0), Replicated()),
            (Partial(), Replicated()),
            (Partial(), Sharded(0)),
        ):
            assert (
                transition_cost(
                    src,
                    dst,
                    message_bytes=1024.0,
                    mesh=mesh,
                    axis_index=0,
                )
                == 0.0
            )

    def test_replicated_to_partial_keeps_one_copy(self) -> None:
        mesh = mesh_1d(4)
        assert not math.isinf(
            transition_cost(
                Replicated(),
                Partial(),
                message_bytes=1.0,
                mesh=mesh,
                axis_index=0,
            )
        )

    def test_2d_mesh_uses_requested_axis_size(self) -> None:
        mesh = DeviceMesh(
            tuple(CPU() for _ in range(2 * 8)), (2, 8), ("dp", "tp")
        )
        on_dp = transition_cost(
            Sharded(0),
            Replicated(),
            message_bytes=1024.0,
            mesh=mesh,
            axis_index=0,
        )
        on_tp = transition_cost(
            Sharded(0),
            Replicated(),
            message_bytes=1024.0,
            mesh=mesh,
            axis_index=1,
        )
        assert math.isclose(on_dp, 1024.0 * _ring_factor(2))
        assert math.isclose(on_tp, 1024.0 * _ring_factor(8))
        assert on_tp > on_dp


class TestUnknownOperand:
    """An Unknown operand skips the rule on its mesh axis."""

    def test_only_the_unknown_row_is_feasible(self) -> None:
        mesh = mesh_1d(2)
        unknown = layout(mesh, (8, 8), (Unknown(),))
        rows = feasible_rows_at_axis(
            [AxisAssignment((R,), (R,))],
            (unknown,),
            mesh,
            0,
            [unknown.placements],
        )
        assert rows == (AxisAssignment((Unknown(),), (Unknown(),)),)

    def test_partial_meeting_unknown_raises(self) -> None:
        mesh = mesh_1d(2)
        unknown = layout(mesh, (8, 8), (Unknown(),))
        partial = layout(mesh, (8, 8), (Partial(),))
        with pytest.raises(ShardingError, match="Resolve Partial"):
            feasible_rows_at_axis(
                [AxisAssignment((R, R), (R,))],
                (unknown, partial),
                mesh,
                0,
                [unknown.placements, partial.placements],
            )
