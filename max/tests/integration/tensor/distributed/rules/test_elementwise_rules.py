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

"""Pure-metadata tests for elementwise placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping
from max.experimental.sharding.rules.elementwise import (
    binary_elementwise,
    ternary_elementwise,
    unary_passthrough,
)
from max.experimental.sharding.types import TensorLayout

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Unary passthrough
# ═════════════════════════════════════════════════════════════════════════


class TestUnaryPassthrough:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (R,)

    def test_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (S(0),)

    def test_sharded_axis1(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (S(1),)

    def test_partial(self) -> None:
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (R,)

    def test_2d_mesh(self) -> None:
        layout = _layout(M(MESH_2D, S(0), R), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (S(0), R)

    def test_2d_mesh_both_sharded(self) -> None:
        layout = _layout(M(MESH_2D, S(0), S(1)), (4, 8))
        _, (out,) = unary_passthrough(layout)
        assert out.to_placements() == (S(0), S(1))


# ═════════════════════════════════════════════════════════════════════════
#  Binary elementwise
# ═════════════════════════════════════════════════════════════════════════


class TestBinaryElementwise:
    def test_both_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        assert out.to_placements() == (R,)

    def test_both_sharded_same(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_sharded_plus_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_replicated_plus_sharded(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        assert out.to_placements() == (S(1),)

    def test_incompatible_sharded_raises(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        with pytest.raises(ValueError, match="incompatible"):
            binary_elementwise(lhs, rhs)

    def test_broadcast_rhs_lower_rank(self) -> None:
        """2-D activation S(1) + 1-D bias S(0): bias S(0) shifts to S(1), matches."""
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8,))
        _, (out,) = binary_elementwise(lhs, rhs)
        # rhs S(0) shifted to S(1), matches lhs S(1)
        assert out.to_placements() == (S(1),)

    def test_broadcast_lhs_lower_rank(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (8,))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        # lhs S(0) shifted to S(1), matches rhs S(1)
        assert out.to_placements() == (S(1),)

    def test_2d_mesh(self) -> None:
        lhs = _layout(M(MESH_2D, S(0), R), (4, 8))
        rhs = _layout(M(MESH_2D, R, S(1)), (4, 8))
        _, (out,) = binary_elementwise(lhs, rhs)
        assert out.to_placements() == (S(0), S(1))


# ═════════════════════════════════════════════════════════════════════════
#  Ternary elementwise
# ═════════════════════════════════════════════════════════════════════════


class TestTernaryElementwise:
    def test_all_replicated(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        assert out.to_placements() == (R,)

    def test_all_sharded_same(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        c = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        assert out.to_placements() == (S(0),)

    def test_cond_sharded_values_replicated(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        assert out.to_placements() == (S(0),)

    def test_values_sharded_cond_replicated(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        c = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        assert out.to_placements() == (S(1),)

    def test_one_value_sharded(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        assert out.to_placements() == (S(0),)

    def test_incompatible_raises(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        with pytest.raises(ValueError, match="incompatible"):
            ternary_elementwise(a, b, c)

    def test_broadcast_rank_adjustment(self) -> None:
        """Cond is 1-D, values are 2-D: cond S(0) shifts to S(1)."""
        a = _layout(M(MESH_1D, S(0)), (8,))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = ternary_elementwise(a, b, c)
        # cond S(0) -> S(1) after broadcast adjust, matches x S(1)
        assert out.to_placements() == (S(1),)
