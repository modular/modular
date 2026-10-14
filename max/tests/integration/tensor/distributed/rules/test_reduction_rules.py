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

"""Pure-metadata tests for reduction placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping, TensorLayout
from max.experimental.sharding.action import AxisAssignment
from max.experimental.sharding.placements import Partial, Sharded
from max.experimental.sharding.rules import (
    cumsum_rule,
    linear_reduce_rule,
    reduce_rule,
)
from max.graph import Shape

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S, pick


def _layout(
    mapping: DeviceMapping,
    shape: tuple[int | str, ...],
    dtype: DType = DType.float32,
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


class TestReduceRule:
    """Tests for reduce_rule (nonlinear: softmax, mean, argmax, etc.)."""

    def test_replicated_any_axis(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=0)
        assert out.placements == (R,)

    def test_sharded_reduce_non_sharded_axis(self) -> None:
        """S(0) + reduce axis=1 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=1)
        assert out.placements == (S(0),)

    def test_sharded_reduce_sharded_axis_falls_back(self) -> None:
        """S(0) + reduce axis=0: nonlinear has no Partial path, falls back to R."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=0)
        assert out.placements == (R,)

    def test_negative_axis(self) -> None:
        """axis=-1 on [4, 8] = axis 1, S(0) should pass through."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=-1)
        assert out.placements == (S(0),)

    def test_negative_axis_sharded_falls_back(self) -> None:
        """axis=-1 on [4, 8] with S(1): nonlinear falls back to R for sharded reduce dim."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=-1)
        assert out.placements == (R,)

    def test_partial_resolved_to_sharded(self) -> None:
        """Nonlinear reduce on Partial: P->S(non-reduce-axis) (1x) beats P->R (2x)."""
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=0)
        assert out.placements == (S(1),)

    def test_2d_mesh_sharded_reduce_falls_back_per_axis(self) -> None:
        """2D mesh: S(0) on dp falls back to R; S(1) on tp preserves."""
        layout = _layout(M(MESH_2D, S(0), S(1)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=0)
        assert out.placements == (R, S(1))

    def test_2d_mesh_reduce_non_sharded(self) -> None:
        """2D mesh: S(0) on dp, R on tp, reduce axis=1 -> pass through."""
        layout = _layout(M(MESH_2D, S(0), R), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=1)
        assert out.placements == (S(0), R)

    def test_3d_tensor(self) -> None:
        """[B, S, H] with S(0), reduce axis=2 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=2)
        assert out.placements == (S(0),)


class TestLinearReduceRule:
    """Tests for linear_reduce_rule (sum; cumsum uses cumsum_rule)."""

    def test_partial_passthrough(self) -> None:
        """Linear reduce: Partial passes through (not resolved)."""
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = pick(linear_reduce_rule, layout, axis=0)
        assert out.placements == (P,)

    def test_sharded_non_reduce_axis(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(linear_reduce_rule, layout, axis=1)
        assert out.placements == (S(0),)

    def test_sharded_reduce_axis_produces_partial(self) -> None:
        """Linear reduce on sharded reduce-axis -> Partial(SUM) (free)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(linear_reduce_rule, layout, axis=0)
        assert isinstance(out.placements[0], type(P))


# ═════════════════════════════════════════════════════════════════════════
#  Reductions with symbolic dims
# ═════════════════════════════════════════════════════════════════════════


class TestReduceWithSymbolicDims:
    """Reductions on tensors with symbolic dims must propagate the
    placement correctly. The output shape carries one fewer dim
    (axis removed) but per-rank symbolic naming is preserved on the
    surviving sharded axes."""

    def test_symbolic_unsharded_axis_reduce_unsharded(self) -> None:
        """Reduce over an unsharded static axis with symbolic
        unsharded ``batch`` -> placement preserved."""
        layout = _layout(M(MESH_1D, R), ("batch", 8))
        _, (out,) = pick(reduce_rule, layout, axis=1)
        assert out.placements == (R,)

    def test_symbolic_sharded_axis_reduce_other_axis(self) -> None:
        """``S("batch") + reduce(axis=1)``: sharded symbolic batch
        survives the reduction along the other (static) axis."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 8))
        _, (out,) = pick(reduce_rule, layout, axis=1)
        assert out.placements == (S(0),)

    def test_symbolic_3d_reduce_hidden(self) -> None:
        """``[batch, seq, H] S(0) + reduce(axis=-1)``: hidden axis
        reduced; sharded symbolic batch preserved."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", "seq", 8))
        _, (out,) = pick(reduce_rule, layout, axis=-1)
        assert out.placements == (S(0),)

    def test_symbolic_3d_reduce_seq_under_seq_sharding(self) -> None:
        """Sharding on ``seq`` and reducing over ``seq`` is the
        nonlinear-reduce-on-sharded-axis case: falls back to R."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", "seq", 8))
        _, (out,) = pick(reduce_rule, layout, axis=1)
        assert out.placements == (R,)

    def test_linear_reduce_symbolic_seq_sharding_yields_partial(
        self,
    ) -> None:
        """Linear reduce (sum) over a sharded *symbolic* axis ->
        Partial output."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", "seq", 8))
        _, (out,) = pick(linear_reduce_rule, layout, axis=1)
        assert out.placements == (P,)

    def test_linear_reduce_symbolic_batch_sharding_passes_through(
        self,
    ) -> None:
        """Linear reduce over a non-sharded axis with sharded
        symbolic batch — placement preserved as Sharded(0)."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", "seq", 8))
        _, (out,) = pick(linear_reduce_rule, layout, axis=2)
        assert out.placements == (S(0),)


class TestCumsumRule:
    """Tests for cumsum_rule (dedicated; MXF-833)."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.int32,
            DType.int64,
            DType.uint8,
            DType.uint64,
        ],
    )
    def test_partial_row_present_for_integral(self, dtype: DType) -> None:
        """Integral inputs keep Partial without a move (P -> P row)."""
        layout = _layout(M(MESH_1D, P), (4, 8), dtype)
        rows = cumsum_rule(layout, axis=0)
        assert AxisAssignment((P,), (P,)) in rows
        _, (out,) = pick(cumsum_rule, layout, axis=0)
        assert out.placements == (P,)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.bfloat16, DType.bool],
    )
    def test_partial_row_absent_for_float_and_bool(self, dtype: DType) -> None:
        """Float/bool inputs get no Partial row: a scan reassociates a
        floating-point sum, so resolve-first stays the only path."""
        layout = _layout(M(MESH_1D, P), (4, 8), dtype)
        rows = cumsum_rule(layout, axis=0)
        assert AxisAssignment((P,), (P,)) not in rows
        # No row keeps the Partial input where it is.
        assert all(
            not isinstance(p, Partial)
            for row in rows
            for p in row.needed_inputs
        )

    def test_sharded_non_scan_axis_preserved(self) -> None:
        """S(d) -> S(d) rows cover every non-scan axis, like reduce_rule."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(cumsum_rule, layout, axis=1)
        assert out.placements == (S(0),)

    def test_sharded_scan_axis_still_resolves(self) -> None:
        """S(scan axis) has no passthrough row: replicate first (MXF-834
        boundary untouched)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8), DType.int64)
        rows = cumsum_rule(layout, axis=0)
        assert AxisAssignment((Sharded(0),), (Sharded(0),)) not in rows
        _, (out,) = pick(cumsum_rule, layout, axis=0)
        assert out.placements == (R,)

    def test_2d_mesh_partial_and_sharded_compose(self) -> None:
        """2D mesh: P on one axis, S(non-scan) on the other composes."""
        layout = _layout(M(MESH_2D, P, S(0)), (4, 8), DType.int32)
        _, (out,) = pick(cumsum_rule, layout, axis=1)
        assert out.placements[0] == P
        assert out.placements[1] == S(0)

    def test_exclusive_reverse_flags_do_not_change_placement(self) -> None:
        """The flags are placement-neutral; every variant keeps P."""
        layout = _layout(M(MESH_1D, P), (4, 8), DType.int64)
        for exclusive, reverse in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            _, (out,) = pick(
                cumsum_rule,
                layout,
                axis=1,
                exclusive=exclusive,
                reverse=reverse,
            )
            assert out.placements == (P,)


class TestScanAndMean:
    """A prefix scan and a mean over a sharded axis need the whole axis."""

    def test_mean_gathers_the_reduced_axis(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(reduce_rule, layout, axis=0)
        assert out.placements == (R,)
