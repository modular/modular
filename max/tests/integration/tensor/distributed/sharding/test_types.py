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
"""Tests for DistributedTensorType and DistributedBufferType.

These types live in ``max.experimental.sharding.types`` and describe how
a tensor's global shape maps to per-device local types for graph compilation.
"""

from __future__ import annotations

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    DistributedTensorType,
    Replicated,
    Sharded,
)
from max.graph import BufferType, SymbolicDim, TensorType

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(
        tuple(CPU() for _ in range(rows * cols)), (rows, cols), ("dp", "tp")
    )


class TestDistributedTensorType:
    def test_basic_construction(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        assert dt.dtype == DType.float32
        assert list(dt.shape) == [8, 16]
        assert dt.rank == 2

    def test_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 4
        for lt in local:
            assert isinstance(lt, TensorType)
            assert lt.dtype == DType.float32
            assert list(lt.shape) == [2, 16]

    def test_replicated_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Replicated()])
        for lt in dt.local_types:
            assert list(lt.shape) == [8, 16]

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DistributedTensorType(
            DType.float32, [8, 16], mesh, [Sharded(0), Sharded(1)]
        )
        local = dt.local_types
        assert len(local) == 8
        for lt in local:
            assert list(lt.shape) == [4, 4]

    def test_symbolic_dim_produces_renamed_symbolic(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Sharded(0)]
        )
        local = dt.local_types
        assert isinstance(local[0].shape[0], SymbolicDim)
        assert local[0].shape[0].name == "batch_tp"

    def test_symbolic_dim_non_sharded_unchanged(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Replicated()]
        )
        local = dt.local_types
        assert isinstance(local[0].shape[0], SymbolicDim)
        assert local[0].shape[0].name == "batch"

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            DistributedTensorType(
                DType.float32, [8, 16], mesh, [Sharded(0), Replicated()]
            )

    def test_out_of_range_shard_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="out of range"):
            DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(5)])

    def test_uneven_static_dim_raises(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [7, 16], mesh, [Sharded(0)])
        with pytest.raises(ValueError, match="not evenly divisible"):
            _ = dt.local_types

    def test_repr(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        r = repr(dt)
        assert "DistributedTensorType" in r
        assert "float32" in r
        assert "Sharded" in r

    def test_local_types_count_matches_device_count(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedTensorType(DType.float32, [8, 4], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 2
        for lt in local:
            assert lt.device is not None


class TestDistributedBufferType:
    def test_basic_construction(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        assert dt.dtype == DType.float32
        assert dt.rank == 2

    def test_local_types_are_buffer_types(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 2
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [4, 4]

    def test_replicated_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(DType.float32, [8, 16], mesh, [Replicated()])
        local = dt.local_types
        assert len(local) == 4
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [8, 16]

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DistributedBufferType(
            DType.float32, [8, 16], mesh, [Sharded(0), Sharded(1)]
        )
        local = dt.local_types
        assert len(local) == 8
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [4, 4]

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            DistributedBufferType(
                DType.float32, [8, 16], mesh, [Sharded(0), Replicated()]
            )

    def test_out_of_range_shard_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="out of range"):
            DistributedBufferType(DType.float32, [8, 16], mesh, [Sharded(5)])

    def test_uneven_static_dim_raises(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(DType.float32, [7, 16], mesh, [Sharded(0)])
        with pytest.raises(ValueError, match="not evenly divisible"):
            _ = dt.local_types

    def test_symbolic_dim_produces_renamed_symbolic(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Sharded(0)]
        )
        local = dt.local_types
        assert isinstance(local[0].shape[0], SymbolicDim)
        assert local[0].shape[0].name == "batch_tp"

    def test_rank_property(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4, 3], mesh, [Sharded(0)])
        assert dt.rank == 3

    def test_dtype_preserved(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.bfloat16, [8, 4], mesh, [Sharded(0)])
        assert dt.dtype == DType.bfloat16

    def test_repr(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        assert "DistributedBufferType" in repr(dt)
