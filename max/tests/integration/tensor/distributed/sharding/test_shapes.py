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
"""Tests for shard shape helpers:
``local_shard_shape_from_global``, ``even_shard_sizes``."""

from __future__ import annotations

import pytest
from max.driver import CPU, Device
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)
from max.experimental.sharding.placements import (
    even_shard_sizes,
    local_shard_shape_from_global,
)
from max.graph import Dim, Shape, StaticDim, SymbolicDim


def cpu_devices(n: int) -> tuple[Device, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(cpu_devices(n), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(cpu_devices(rows * cols), (rows, cols), ("dp", "tp"))


class TestEvenShardSizes:
    def test_even_split(self) -> None:
        assert even_shard_sizes(8, 4) == [2, 2, 2, 2]

    def test_uneven_split(self) -> None:
        assert even_shard_sizes(10, 4) == [3, 3, 2, 2]

    def test_minus_one_preserved(self) -> None:
        """``-1`` (reshape wildcard) propagates verbatim per shard so each
        rank's local reshape can infer its own extent from the actual
        local input."""
        assert even_shard_sizes(-1, 4) == [-1, -1, -1, -1]

    def test_single_shard(self) -> None:
        assert even_shard_sizes(8, 1) == [8]


# ═════════════════════════════════════════════════════════════════════════
#  local_shard_shape_from_global
# ═════════════════════════════════════════════════════════════════════════


class TestLocalShardShapeFromGlobal:
    def test_even_1d(self) -> None:
        mesh = mesh_1d(4)
        shapes = local_shard_shape_from_global(
            Shape([8, 4]), mesh, [Sharded(0)]
        )
        assert len(shapes) == 4
        for s in shapes:
            assert list(s) == [StaticDim(2), StaticDim(4)]

    def test_uneven_1d(self) -> None:
        mesh = mesh_1d(4)
        shapes = local_shard_shape_from_global(
            Shape([10, 4]), mesh, [Sharded(0)]
        )
        assert len(shapes) == 4
        assert int(shapes[0][0]) == 3
        assert int(shapes[1][0]) == 3
        assert int(shapes[2][0]) == 2
        assert int(shapes[3][0]) == 2

    def test_replicated(self) -> None:
        mesh = mesh_1d(4)
        shapes = local_shard_shape_from_global(
            Shape([8, 4]), mesh, [Replicated()]
        )
        for s in shapes:
            assert list(s) == [StaticDim(8), StaticDim(4)]

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 2)
        shapes = local_shard_shape_from_global(
            Shape([8, 12]), mesh, [Sharded(0), Sharded(1)]
        )
        assert len(shapes) == 4
        for s in shapes:
            assert list(s) == [StaticDim(4), StaticDim(6)]

    def test_placement_count_mismatch_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            local_shard_shape_from_global(
                Shape([8]), mesh, [Sharded(0), Replicated()]
            )

    def test_out_of_range_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="out of range"):
            local_shard_shape_from_global(Shape([8]), mesh, [Sharded(5)])

    def test_partial_leaves_shape_unchanged(self) -> None:
        mesh = mesh_1d(4)
        shapes = local_shard_shape_from_global(Shape([8, 4]), mesh, [Partial()])
        for s in shapes:
            assert list(s) == [StaticDim(8), StaticDim(4)]

    def test_2d_mesh_mixed_shard_replicate(self) -> None:
        mesh = mesh_2d(2, 4)
        shapes = local_shard_shape_from_global(
            Shape([8, 12]), mesh, [Sharded(0), Replicated()]
        )
        assert len(shapes) == 8
        for s in shapes:
            assert list(s) == [StaticDim(4), StaticDim(12)]

    def test_symbolic_per_rank_dim_emits_fresh_symbolic(self) -> None:
        """SymbolicDim on a Sharded axis → fresh ``{name}_{axis_name}_{coord}``
        per-rank symbol. Replicated axes pass through unchanged."""
        mesh = mesh_1d(2)
        shapes = local_shard_shape_from_global(
            Shape([Dim("batch"), Dim(4)]), mesh, [Sharded(0)]
        )
        assert len(shapes) == 2
        for rank, s in enumerate(shapes):
            assert list(s) == [SymbolicDim(f"batch_tp_{rank}"), StaticDim(4)]

    def test_negative_axis_wraps(self) -> None:
        mesh = mesh_1d(2)
        shapes = local_shard_shape_from_global(
            Shape([8, 4]), mesh, [Sharded(-1)]
        )
        for s in shapes:
            assert list(s) == [StaticDim(8), StaticDim(2)]

    def test_3d_tensor(self) -> None:
        mesh = mesh_1d(2)
        shapes = local_shard_shape_from_global(
            Shape([4, 6, 8]), mesh, [Sharded(2)]
        )
        assert len(shapes) == 2
        for s in shapes:
            assert list(s) == [StaticDim(4), StaticDim(6), StaticDim(4)]
