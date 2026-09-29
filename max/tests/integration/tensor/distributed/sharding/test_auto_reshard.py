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
"""Tests for :func:`max.experimental.sharding.auto_reshard` and the picker."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    ActionSet,
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
    build_action_set,
)
from max.experimental.sharding._auto_reshard import (
    _AUTO_RESHARD_POLICY,
    pick_reshard_action,
)
from max.experimental.sharding.action import Action


def policy() -> tuple[frozenset[Transition], str]:
    return _AUTO_RESHARD_POLICY.get()


def pick(action_set: ActionSet, operand_names: Sequence[str] = ()) -> Action:
    return pick_reshard_action(
        action_set, op_name="custom_op", operand_names=operand_names
    )


def mesh_1d(n: int) -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), ("tp",))


def mesh_2d() -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(4)), (2, 2), ("dp", "tp"))


def layout(
    mesh: DeviceMesh, shape: tuple[int, ...], *placements: Placement
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placements))


def inputs(action: Action, slot: int = 0) -> tuple[Placement, ...]:
    return action.inputs[slot].placements


def action_set(lay: TensorLayout, *rows: AxisAssignment) -> ActionSet:
    return build_action_set(rows, layouts=(lay,))


def action_set_without_fallback(
    lay: TensorLayout, *rows: AxisAssignment
) -> ActionSet:
    """Builds an action set with no ``(R,) -> R`` row: every row moves."""
    return ActionSet(axis_assignments=rows, layouts=(lay,), mesh=lay.mesh)


R, S0, S1, P = Replicated(), Sharded(0), Sharded(1), Partial()


@pytest.fixture
def partial_input() -> ActionSet:
    """Builds an action set whose rows allreduce or reduce-scatter a Partial."""
    lay = layout(mesh_1d(4), (1024,), P)
    return action_set(lay, AxisAssignment((R,), R), AxisAssignment((S0,), S0))


class TestRaiseMode:
    def test_a_row_that_moves_an_input_raises(
        self, partial_input: ActionSet
    ) -> None:
        with (
            auto_reshard(mode="raise"),
            pytest.raises(ShardingError, match=r"x\{P->R\}.*allreduce\(x\)"),
        ):
            pick(partial_input, ("x",))

    def test_a_free_local_slice_raises_too(self) -> None:
        lay = layout(mesh_1d(4), (16,), R)
        with (
            auto_reshard(mode="raise"),
            pytest.raises(ShardingError, match=r"arg0\{R->S0\}.*local_slice"),
        ):
            pick(action_set_without_fallback(lay, AxisAssignment((S0,), S0)))

    def test_a_single_device_input_reads_as_its_own_mesh(self) -> None:
        """Its placement is whatever its mapping says, not a special case."""
        lay = layout(mesh_1d(2), (16,), S0)
        single = layout(DeviceMesh.single(CPU()), (16,), R)
        s = build_action_set(
            [AxisAssignment((S0, R), S0)], layouts=(lay, single)
        )
        picked = pick(s)
        assert inputs(picked, 0) == (S0,)
        assert inputs(picked, 1) == (R,)

    def test_an_input_on_another_mesh_counts_as_a_move(self) -> None:
        """Landing on the op's mesh is a move, whatever its placement there."""
        lay = layout(mesh_1d(2), (16,), R)
        other = layout(DeviceMesh((CPU(), CPU()), (2,), ("dp",)), (16,), R)
        s = build_action_set([AxisAssignment((R, R), R)], layouts=(lay, other))
        with (
            auto_reshard(mode="raise"),
            pytest.raises(ShardingError, match="arg1"),
        ):
            pick(s)
        with auto_reshard(mode="silent"):
            pick(s)

    def test_a_row_that_moves_nothing_runs_silently(self) -> None:
        lay = layout(mesh_1d(4), (16,), S0)
        with auto_reshard(mode="raise"), warnings.catch_warnings():
            warnings.simplefilter("error")
            picked = pick(action_set(lay, AxisAssignment((S0,), S0)))
        assert inputs(picked) == (S0,)

    def test_the_message_shows_the_decision_and_both_ways_out(
        self, partial_input: ActionSet
    ) -> None:
        with (
            auto_reshard(mode="raise"),
            pytest.raises(ShardingError) as info,
        ):
            pick(partial_input, ("x",))
        text = str(info.value)
        assert "custom_op(x) needs a resharding" in text
        assert (
            'region : auto_reshard(DEFAULT_TRANSITIONS, mode="raise")' in text
        )
        cheaper, chosen = [l for l in text.splitlines() if "x{P->" in l]
        assert "x{P->S0}" in cheaper and cheaper.endswith("not allowed")
        assert "x{P->R}" in chosen and chosen.endswith("chosen, reshards x")
        assert "  x = transfer_to(x, DeviceMapping(mesh, (R,)))" in text
        assert '  with auto_reshard(mode="silent"):' in text
        assert 'with auto_reshard(ALL_TRANSITIONS, mode="silent")' in text


class TestRanking:
    def test_no_movement_beats_a_free_local_slice(self) -> None:
        """Both rows cost 0; the one that leaves the input alone wins."""
        lay = layout(mesh_1d(4), (16, 16), R)
        s = action_set(lay, AxisAssignment((S0,), S0), AxisAssignment((R,), R))
        assert inputs(pick(s)) == (R,)

    def test_passthrough_wins_even_when_declared_last(self) -> None:
        lay = layout(mesh_1d(4), (16,), S0)
        s = action_set(lay, AxisAssignment((R,), R), AxisAssignment((S0,), S0))
        assert inputs(pick(s)) == (S0,)

    def test_equal_cost_and_moves_fall_back_to_declaration_order(
        self,
    ) -> None:
        lay = layout(mesh_1d(4), (16, 16), R)
        s0, s1 = AxisAssignment((S0,), S0), AxisAssignment((S1,), S1)
        assert inputs(pick(action_set_without_fallback(lay, s0, s1))) == (S0,)
        assert inputs(pick(action_set_without_fallback(lay, s1, s0))) == (S1,)


class TestReshardMode:
    def test_silent_applies_the_cheapest_allowed_row(
        self, partial_input: ActionSet
    ) -> None:
        with auto_reshard(mode="silent"), warnings.catch_warnings():
            warnings.simplefilter("error")
            picked = pick(partial_input)
        assert inputs(picked) == (R,), (
            "allreduce: reduce_scatter is not default"
        )
        with auto_reshard(ALL_TRANSITIONS, mode="silent"):
            assert inputs(pick(partial_input)) == (S0,), (
                "reduce_scatter is cheaper"
            )

    def test_warn_applies_it_and_says_so(
        self, partial_input: ActionSet
    ) -> None:
        with (
            auto_reshard(mode="warn"),
            pytest.warns(UserWarning, match=r"arg0\{P->R\}"),
        ):
            picked = pick(partial_input)
        assert inputs(picked) == (R,)

    def test_warn_is_quiet_when_nothing_moves(self) -> None:
        lay = layout(mesh_1d(4), (16,), S0)
        with auto_reshard(mode="warn"), warnings.catch_warnings():
            warnings.simplefilter("error")
            pick(action_set(lay, AxisAssignment((S0,), S0)))


class TestAllowedTransitions:
    def test_narrowing_changes_the_pick(self, partial_input: ActionSet) -> None:
        with auto_reshard({(Partial, Replicated)}, mode="silent"):
            assert inputs(pick(partial_input)) == (R,)
        with auto_reshard({(Partial, Sharded)}, mode="silent"):
            assert inputs(pick(partial_input)) == (S0,)

    def test_no_allowed_row_raises_and_names_what_to_allow(
        self, partial_input: ActionSet
    ) -> None:
        with (
            auto_reshard({(Sharded, Replicated)}, mode="silent"),
            pytest.raises(ShardingError) as info,
        ):
            pick(partial_input)
        text = str(info.value)
        assert (
            "no resharding on mesh axis 'tp' uses only the transitions "
            "this region allows" in text
        )
        assert "chosen" not in text
        assert text.count("not allowed") == 2
        assert (
            "with auto_reshard({(Partial, Replicated), "
            '(Sharded, Replicated)}, mode="silent"):' in text
        ), "the recommendation stays within DEFAULT_TRANSITIONS"

    def test_allowing_a_transition_does_not_silence_it(
        self, partial_input: ActionSet
    ) -> None:
        with (
            auto_reshard({(Partial, Replicated)}, mode="raise"),
            pytest.raises(ShardingError, match=r"arg0\{P->R\}"),
        ):
            pick(partial_input)


class TestScoping:
    def test_default_outside_any_block(self) -> None:
        assert policy() == (DEFAULT_TRANSITIONS, "silent")
        assert DEFAULT_TRANSITIONS == ALL_TRANSITIONS - {(Partial, Sharded)}

    def test_a_bare_block_is_the_default_policy(self) -> None:
        with auto_reshard():
            assert policy() == (DEFAULT_TRANSITIONS, "silent")

    def test_a_nested_block_keeps_what_it_leaves_out(self) -> None:
        with auto_reshard({(Partial, Replicated)}, mode="raise"):
            with auto_reshard(mode="warn"):
                assert policy() == ({(Partial, Replicated)}, "warn")
            with auto_reshard(ALL_TRANSITIONS):
                assert policy() == (ALL_TRANSITIONS, "raise")
            assert policy() == ({(Partial, Replicated)}, "raise")
        assert policy() == (DEFAULT_TRANSITIONS, "silent")

    def test_an_unknown_mode_is_rejected(self) -> None:
        unknown: Any = "loud"
        with pytest.raises(ValueError, match="mode must be one of"):
            with auto_reshard(mode=unknown):
                pass

    def test_a_bare_pair_is_rejected(self) -> None:
        bare: Any = (Partial, Replicated)
        with pytest.raises(TypeError, match="pairs of placement types"):
            with auto_reshard(bare):
                pass

    def test_an_exception_still_restores(self) -> None:
        with pytest.raises(RuntimeError):
            with auto_reshard(mode="warn"):
                raise RuntimeError("boom")
        assert policy() == (DEFAULT_TRANSITIONS, "silent")

    def test_works_as_a_decorator(self) -> None:
        @auto_reshard(mode="warn")
        def f() -> tuple[frozenset[Transition], str]:
            return policy()

        assert f() == (DEFAULT_TRANSITIONS, "warn")
        assert policy() == (DEFAULT_TRANSITIONS, "silent")


class TestPerMeshAxis:
    def test_an_infeasible_row_is_skipped(self) -> None:
        """Extent 1 cannot be sharded, so the later-declared S1 row wins."""
        lay = layout(mesh_1d(2), (1, 8), R)
        s = action_set_without_fallback(
            lay, AxisAssignment((S0,), S0), AxisAssignment((S1,), S1)
        )
        assert inputs(pick(s)) == (S1,)

    def test_an_unreachable_row_is_skipped(self) -> None:
        """Nothing turns a Replicated input into a Partial one."""
        lay = layout(mesh_1d(2), (16,), R)
        s = action_set_without_fallback(
            lay, AxisAssignment((P,), P), AxisAssignment((R,), R)
        )
        assert inputs(pick(s)) == (R,)

    def test_a_2d_mesh_gets_one_row_per_axis(self) -> None:
        lay = layout(mesh_2d(), (16, 16), S0, S1)
        s = action_set(
            lay, AxisAssignment((S0,), S0), AxisAssignment((S1,), S1)
        )
        assert inputs(pick(s)) == (S0, S1)

    def test_a_committed_axis_constrains_the_next(self) -> None:
        """Extent 2 admits one mesh axis of size 2, not both."""
        lay = layout(mesh_2d(), (2, 16), R, R)
        s = action_set_without_fallback(
            lay, AxisAssignment((S0,), S0), AxisAssignment((S1,), S1)
        )
        assert inputs(pick(s)) == (S0, S1)

    def test_no_reachable_row_raises(self) -> None:
        lay = layout(mesh_1d(2), (16,), R)
        with pytest.raises(ShardingError, match="no feasible plan"):
            pick(ActionSet(axis_assignments=(), layouts=(lay,), mesh=lay.mesh))
