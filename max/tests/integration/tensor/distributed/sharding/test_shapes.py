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
"""Tests for shard shape math functions: shard_shape,
global_shape_from_local, local_shard_shape_from_global,
_shard_sizes_along_axis, and symbolic dims."""

from __future__ import annotations

import pytest
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
    _shard_sizes_along_axis,
    global_shape_from_local,
    local_shard_shape_from_global,
    shard_shape,
)
from max.experimental.tensor import Tensor
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


class TestGlobalShapeFromLocalShards:
    def test_basic(self) -> None:
        mesh = mesh_1d(4)
        local = [[Dim(2), Dim(4)]] * 4
        result = global_shape_from_local(local, mesh, [Sharded(0)])
        assert list(result) == [StaticDim(8), StaticDim(4)]

    def test_replicated(self) -> None:
        mesh = mesh_1d(4)
        local = [[Dim(8), Dim(4)]] * 4
        result = global_shape_from_local(local, mesh, [Replicated()])
        assert list(result) == [StaticDim(8), StaticDim(4)]

    def test_partial(self) -> None:
        mesh = mesh_1d(4)
        local = [[Dim(8)]] * 4
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

    def test_symbolic_sharded_dim_emits_per_coord_symbolic(self) -> None:
        """SymbolicDim on a Sharded axis → per-coord-distinct symbolic dim.

        Each device gets a name ``{parent}_{axis_name}_{coord}`` where
        ``coord`` is the device's coordinate along the sharded mesh
        axis. Distinct names per coord prevent the graph's
        ``same name = same size`` convention from collapsing per-shard
        runtime sizes (which is exactly what data parallelism with
        uneven batches needs).
        """
        mesh = mesh_1d(2)  # axis name defaults to "tp"
        shapes = local_shard_shape_from_global(
            Shape([Dim("batch"), Dim(4)]), mesh, [Sharded(0)]
        )
        assert len(shapes) == 2
        assert list(shapes[0]) == [SymbolicDim("batch_tp_0"), StaticDim(4)]
        assert list(shapes[1]) == [SymbolicDim("batch_tp_1"), StaticDim(4)]

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


class TestGlobalShapeFromLocalShardsEdgeCases:
    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        local = [[Dim(4), Dim(3)]] * mesh.num_devices
        result = global_shape_from_local(local, mesh, [Sharded(0), Sharded(1)])
        assert list(result) == [StaticDim(8), StaticDim(12)]

    def test_unknown_placement_raises(self) -> None:
        mesh = mesh_1d(2)

        class CustomPlacement:
            pass

        with pytest.raises(NotImplementedError, match="Unknown placement"):
            global_shape_from_local(
                [[Dim(4)]] * mesh.num_devices,
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
        local = [[Dim(2), Dim(3), Dim(8)]] * 4
        result = global_shape_from_local(local, mesh, [Sharded(0)])
        assert list(result) == [StaticDim(8), StaticDim(3), StaticDim(8)]


# ═════════════════════════════════════════════════════════════════════════
#  global_shape_from_local with reference_tensors
#
#  The base behaviour (no references — sum static, multiply rest) is
#  covered above. These tests focus on the symbolic-dim recovery path:
#  when a sharded ``SymbolicDim`` per-shard name appears in a reference
#  tensor's shape table, the function recovers the global symbolic dim
#  instead of falling back to multiplication.
# ═════════════════════════════════════════════════════════════════════════


def _make_sharded(
    per_shard_shape: list[int],
    global_shape: list[int],
    mesh: DeviceMesh,
    placements: tuple,  # type: ignore[type-arg]
) -> Tensor:
    """Build a realized sharded Tensor with explicit per-shard and global shapes."""
    bufs = tuple(
        Buffer.zeros(per_shard_shape, dtype=DType.float32, device=CPU())
        for _ in range(mesh.num_devices)
    )
    return Tensor._from_shards(
        bufs, mesh, placements, global_shape=global_shape
    )


class TestStaticDimsIgnoreReferences:
    """For static dims, summing per-device shard sizes is the answer.
    References are never consulted, so collisions like "the same
    ``StaticDim(2)`` showing up on a sharded and a replicated reference"
    can't happen.
    """

    def test_sum_works_without_references(self) -> None:
        mesh = mesh_1d(4)
        local = [[Dim(2), Dim(8)]] * 4
        result = global_shape_from_local(
            local, mesh, [Sharded(0)], reference_tensors=()
        )
        assert list(result) == [StaticDim(8), StaticDim(8)]

    def test_static_collision_does_not_use_references(self) -> None:
        """Same StaticDim on two references with conflicting "global"
        values — the function ignores references for static dims and
        returns the sum."""
        mesh = mesh_1d(2)
        replicated = _make_sharded([2, 3], [2, 3], mesh, (Replicated(),))
        sharded = _make_sharded([1, 2], [1, 4], mesh, (Sharded(1),))
        local = [[Dim(2), Dim(2)], [Dim(2), Dim(2)]]
        result = global_shape_from_local(
            local,
            mesh,
            [Sharded(1)],
            reference_tensors=[replicated, sharded],
        )
        # Axis 0 (non-sharded) passes through: 2.
        # Axis 1 (sharded, static): 2 + 2 = 4.
        assert list(result) == [StaticDim(2), StaticDim(4)]


class TestSymbolicDimRecovery:
    def test_per_device_symbolic_recovers_global_name(self) -> None:
        """Per-device-distinct symbolic shard names (``batch_dp_0``,
        ``batch_dp_1``, …) all map back to the same global symbolic dim."""
        with F.lazy():
            mesh = mesh_1d(2, name="dp")
            mapping = PlacementMapping(mesh, (Sharded(0),))
            ref = Tensor.zeros(
                ["batch", 8], dtype=DType.float32, device=mapping
            )

            shard0 = ref.local_shards[0].shape[0]
            shard1 = ref.local_shards[1].shape[0]
            assert shard0 != shard1  # per-coord names are distinct

            result = global_shape_from_local(
                [[shard0, Dim(8)], [shard1, Dim(8)]],
                mesh,
                [Sharded(0)],
                reference_tensors=[ref],
            )

        assert list(result) == [SymbolicDim("batch"), StaticDim(8)]

    def test_unknown_symbolic_falls_back_to_multiplication(self) -> None:
        """A sharded SymbolicDim with no matching reference becomes
        ``dim * mesh_axis_size`` (lossy AlgebraicDim). The function
        always returns a Shape — never None."""
        with F.lazy():
            mesh = mesh_1d(2, name="dp")
            mapping = PlacementMapping(mesh, (Sharded(0),))
            ref = Tensor.zeros(
                ["batch", 8], dtype=DType.float32, device=mapping
            )

            unknown = SymbolicDim("unrelated_dp_0")
            result = global_shape_from_local(
                [[unknown, Dim(8)], [unknown, Dim(8)]],
                mesh,
                [Sharded(0)],
                reference_tensors=[ref],
            )

        assert list(result)[1] == StaticDim(8)
        # The sharded axis is the lossy fallback — symbolic identity is lost.
        assert list(result)[0] != SymbolicDim("unrelated_dp_0")

    def test_non_distributed_reference_is_skipped(self) -> None:
        """Single-device tensors aren't distributed and contribute nothing
        to the symbolic lookup table."""
        single = Tensor.zeros([4, 8], dtype=DType.float32, device=CPU())
        assert not single.is_distributed

        with F.lazy():
            mesh = mesh_1d(2, name="dp")
            mapping = PlacementMapping(mesh, (Sharded(0),))
            ref = Tensor.zeros(
                ["batch", 8], dtype=DType.float32, device=mapping
            )
            shard0 = ref.local_shards[0].shape[0]
            shard1 = ref.local_shards[1].shape[0]
            # Only the non-distributed tensor as reference → skipped.
            result = global_shape_from_local(
                [[shard0, Dim(8)], [shard1, Dim(8)]],
                mesh,
                [Sharded(0)],
                reference_tensors=[single],
            )

        assert list(result)[0] != SymbolicDim("batch")

    def test_lookup_finds_match_across_multiple_references(self) -> None:
        """Walks every reference tensor populating the symbolic-dim table."""
        with F.lazy():
            mesh = mesh_1d(2, name="dp")
            mapping = PlacementMapping(mesh, (Sharded(0),))
            ref_a = Tensor.zeros(
                ["unrelated", 8], dtype=DType.float32, device=mapping
            )
            ref_b = Tensor.zeros(
                ["batch", 8], dtype=DType.float32, device=mapping
            )

            shard0 = ref_b.local_shards[0].shape[0]
            shard1 = ref_b.local_shards[1].shape[0]
            result = global_shape_from_local(
                [[shard0, Dim(8)], [shard1, Dim(8)]],
                mesh,
                [Sharded(0)],
                reference_tensors=[ref_a, ref_b],
            )

        assert list(result) == [SymbolicDim("batch"), StaticDim(8)]
