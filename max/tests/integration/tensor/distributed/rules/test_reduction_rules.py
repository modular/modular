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
from max.experimental.sharding import DeviceMapping
from max.experimental.sharding.rules.reduction import (
    linear_reduce_rule,
    reduce_rule,
)
from max.experimental.sharding.types import TensorLayout

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


class TestReduceRule:
    """Tests for reduce_rule (nonlinear: softmax, mean, argmax, etc.)."""

    def test_replicated_any_axis(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = reduce_rule(layout, axis=0)
        assert out.to_placements() == (R,)

    def test_sharded_reduce_non_sharded_axis(self) -> None:
        """S(0) + reduce axis=1 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = reduce_rule(layout, axis=1)
        assert out.to_placements() == (S(0),)

    def test_sharded_reduce_sharded_axis_raises(self) -> None:
        """S(0) + reduce axis=0 -> error."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="sharded axis"):
            reduce_rule(layout, axis=0)

    def test_negative_axis(self) -> None:
        """axis=-1 on [4, 8] = axis 1, S(0) should pass through."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = reduce_rule(layout, axis=-1)
        assert out.to_placements() == (S(0),)

    def test_negative_axis_sharded_raises(self) -> None:
        """axis=-1 on [4, 8] with S(1) -> error (reducing sharded dim)."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        with pytest.raises(ValueError, match="sharded axis"):
            reduce_rule(layout, axis=-1)

    def test_partial_resolved_to_replicated(self) -> None:
        """Nonlinear reduce resolves Partial -> Replicated."""
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = reduce_rule(layout, axis=0)
        assert out.to_placements() == (R,)

    def test_2d_mesh_sharded_reduce_raises(self) -> None:
        """2D mesh: S(0) on dp, S(1) on tp, reduce axis=0 -> error on dp."""
        layout = _layout(M(MESH_2D, S(0), S(1)), (4, 8))
        with pytest.raises(ValueError, match="sharded axis"):
            reduce_rule(layout, axis=0)

    def test_2d_mesh_reduce_non_sharded(self) -> None:
        """2D mesh: S(0) on dp, R on tp, reduce axis=1 -> pass through."""
        layout = _layout(M(MESH_2D, S(0), R), (4, 8))
        _, (out,) = reduce_rule(layout, axis=1)
        assert out.to_placements() == (S(0), R)

    def test_3d_tensor(self) -> None:
        """[B, S, H] with S(0), reduce axis=2 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        _, (out,) = reduce_rule(layout, axis=2)
        assert out.to_placements() == (S(0),)


class TestLinearReduceRule:
    """Tests for linear_reduce_rule (sum, cumsum)."""

    def test_partial_passthrough(self) -> None:
        """Linear reduce: Partial passes through (not resolved)."""
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = linear_reduce_rule(layout, axis=0)
        assert out.to_placements() == (P,)

    def test_sharded_non_reduce_axis(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = linear_reduce_rule(layout, axis=1)
        assert out.to_placements() == (S(0),)

    def test_sharded_reduce_axis_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="sharded axis"):
            linear_reduce_rule(layout, axis=0)
