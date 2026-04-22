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
"""Tests for shard shape math functions: shard_shape, global_shape_from_local,
local_shard_shape_from_global, _shard_sizes_along_axis, and symbolic dims."""

from __future__ import annotations

import pytest
from max.driver import CPU, Device
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    _shard_sizes_along_axis,
    global_shape_from_local,
    local_shard_shape_from_global,
    shard_shape,
)
from max.graph import Dim, Shape, StaticDim, SymbolicDim

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def cpu_devices(n: int) -> tuple[Device, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(cpu_devices(n), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(cpu_devices(rows * cols), (rows, cols), ("dp", "tp"))


# ═════════════════════════════════════════════════════════════════════════
#  Shard shape math
# ═════════════════════════════════════════════════════════════════════════


class TestShardShape:
    def test_shard_shape_1d(self) -> None:
        result = shard_shape([Dim(8), Dim(4)], [Sharded(0)], [4])
        assert [int(d) for d in result] == [2, 4]

    def test_shard_shape_replicated(self) -> None:
        result = shard_shape([Dim(8), Dim(4)], [Replicated()], [4])
        assert [int(d) for d in result] == [8, 4]

    def test_shard_shape_2d_mesh(self) -> None:
        result = shard_shape(
            [Dim(8), Dim(12)], [Sharded(0), Sharded(1)], [2, 4]
        )
        assert [int(d) for d in result] == [4, 3]

    def test_shard_shape_partial_no_change(self) -> None:
        result = shard_shape([Dim(8), Dim(4)], [Partial()], [4])
        assert [int(d) for d in result] == [8, 4]


class TestGlobalShapeFromLocal:
    def test_basic(self) -> None:
        mesh = mesh_1d(4)
        local = [Dim(2), Dim(4)]
        result = global_shape_from_local(local, mesh, [Sharded(0)])
        assert list(result) == [StaticDim(8), StaticDim(4)]

    def test_replicated(self) -> None:
        mesh = mesh_1d(4)
        local = [Dim(8), Dim(4)]
        result = global_shape_from_local(local, mesh, [Replicated()])
        assert list(result) == [StaticDim(8), StaticDim(4)]

    def test_partial(self) -> None:
        mesh = mesh_1d(4)
        local = [Dim(8)]
        result = global_shape_from_local(local, mesh, [Partial()])
        assert list(result) == [StaticDim(8)]


class TestShardSizesAlongAxis:
    def test_even_split(self) -> None:
        assert _shard_sizes_along_axis(8, 4) == [2, 2, 2, 2]

    def test_uneven_split(self) -> None:
        assert _shard_sizes_along_axis(10, 4) == [3, 3, 2, 2]

    def test_single_shard(self) -> None:
        assert _shard_sizes_along_axis(5, 1) == [5]

    def test_more_shards_than_elements(self) -> None:
        assert _shard_sizes_along_axis(2, 4) == [1, 1, 0, 0]


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
        # 10 / 4 = [3, 3, 2, 2]
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
        mesh = mesh_2d(2, 2)  # 4 devices total
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

    def test_symbolic_sharded_dim_emits_fresh_symbolic(self) -> None:
        """SymbolicDim on a Sharded axis → fresh ``{name}_{axis_name}`` dim.

        Mirrors the graph-layer behaviour in
        :meth:`DistributedTensorType._local_shard_shape`. Replicated axes
        pass through unchanged.
        """
        mesh = mesh_1d(2)  # axis name defaults to "tp"
        shapes = local_shard_shape_from_global(
            Shape([Dim("batch"), Dim(4)]), mesh, [Sharded(0)]
        )
        assert len(shapes) == 2
        for s in shapes:
            assert list(s) == [SymbolicDim("batch_tp"), StaticDim(4)]

    def test_negative_axis_wraps(self) -> None:
        mesh = mesh_1d(2)
        # Sharded(axis=-1) on a rank-2 tensor should shard dim 1
        shapes = local_shard_shape_from_global(
            Shape([8, 4]), mesh, [Sharded(-1)]
        )
        for s in shapes:
            assert list(s) == [StaticDim(8), StaticDim(2)]


# ═════════════════════════════════════════════════════════════════════════
#  Symbolic dimensions in shard math
# ═════════════════════════════════════════════════════════════════════════


class TestSymbolicShardShape:
    def test_shard_shape_with_symbolic_dim(self) -> None:
        # Symbolic dims go through the Dim // int path
        sym = Dim("batch")
        result = shard_shape([sym, Dim(4)], [Sharded(0)], [2])
        # Result is symbolic — just check it's a Dim, not crash
        assert isinstance(result[0], Dim)
        assert int(result[1]) == 4

    def test_shard_shape_replicated_preserves_symbolic(self) -> None:
        sym = Dim("seq")
        result = shard_shape([sym, Dim(4)], [Replicated()], [2])
        assert result[0] == sym


# ═════════════════════════════════════════════════════════════════════════
#  GlobalShapeFromLocal edge cases
# ═════════════════════════════════════════════════════════════════════════


class TestGlobalShapeFromLocalEdgeCases:
    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        local = [Dim(4), Dim(3)]
        result = global_shape_from_local(local, mesh, [Sharded(0), Sharded(1)])
        assert list(result) == [StaticDim(8), StaticDim(12)]

    def test_unknown_placement_raises(self) -> None:
        mesh = mesh_1d(2)

        class CustomPlacement:
            pass

        with pytest.raises(NotImplementedError, match="Unknown placement"):
            global_shape_from_local(
                [Dim(4)],
                mesh,
                [CustomPlacement()],  # type: ignore[list-item]
            )


# ═════════════════════════════════════════════════════════════════════════
#  Higher-rank tensor shard math
# ═════════════════════════════════════════════════════════════════════════


class TestHighRankShardMath:
    def test_shard_shape_3d(self) -> None:
        # Shard dim 1 of a 3D tensor across 4 devices
        result = shard_shape([Dim(2), Dim(8), Dim(16)], [Sharded(1)], [4])
        assert [int(d) for d in result] == [2, 2, 16]

    def test_shard_shape_4d_2d_mesh(self) -> None:
        # 4D tensor, 2D mesh: shard dim 0 on axis 0, dim 2 on axis 1
        result = shard_shape(
            [Dim(8), Dim(4), Dim(12), Dim(3)],
            [Sharded(0), Sharded(2)],
            [2, 4],
        )
        assert [int(d) for d in result] == [4, 4, 3, 3]

    def test_local_shard_shape_3d_tensor(self) -> None:
        mesh = mesh_1d(2)
        shapes = local_shard_shape_from_global(
            Shape([4, 6, 8]), mesh, [Sharded(2)]
        )
        assert len(shapes) == 2
        for s in shapes:
            assert list(s) == [StaticDim(4), StaticDim(6), StaticDim(4)]

    def test_global_from_local_3d(self) -> None:
        mesh = mesh_1d(4)
        local = [Dim(2), Dim(3), Dim(8)]
        result = global_shape_from_local(local, mesh, [Sharded(0)])
        assert list(result) == [StaticDim(8), StaticDim(3), StaticDim(8)]
