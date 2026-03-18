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
"""Tests for DTensor construction, properties, factories, and stubs."""

from __future__ import annotations

import pytest
from conftest import mesh_1d, mesh_2d
from max.driver import CPU
from max.dtype import DType
from max.experimental.distributed import (
    DTensor,
    Partial,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Dim, StaticDim, SymbolicDim


class TestDTensorOnes:
    def test_creates_correct_shard_count(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Replicated()])
        assert len(dt.local_shards) == 4

    def test_replicated_shard_shape(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Replicated()])
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 16]

    def test_sharded_shard_shape(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [2, 16]

    def test_global_shape(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=0)])
        assert dt.shape == [8, 16]

    def test_dtype_default_float32(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 4], mesh, [Replicated()])
        assert dt.dtype == DType.float32

    def test_dtype_explicit(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 4], mesh, [Replicated()], dtype=DType.float64
        )
        assert dt.dtype == DType.float64


class TestDTensorZeros:
    def test_creates_correct_shard_count(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_zeros([6, 4], mesh, [Sharded(axis=0)])
        assert len(dt.local_shards) == 2

    def test_shard_shape(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_zeros([6, 4], mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [3, 4]


class TestDTensorFull:
    def test_custom_value(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_full([4, 4], 42.0, mesh, [Replicated()])
        assert dt.shape == [4, 4]
        assert len(dt.local_shards) == 2

    def test_sharded_divides_dimension(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_full([12, 8], 3.14, mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [3, 8]
        assert dt.shape == [12, 8]

    def test_replicated_keeps_full_shape(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_full([12, 8], 3.14, mesh, [Replicated()])
        for shard in dt.local_shards:
            assert list(shard.shape) == [12, 8]


class TestDTensorFactory2DMesh:
    def test_replicated_replicated(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated(), Replicated()]
        )
        assert dt.shape == [8, 16]
        assert len(dt.local_shards) == 8
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 16]

    def test_replicated_sharded(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated(), Sharded(axis=1)]
        )
        assert dt.shape == [8, 16]
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 4]

    def test_sharded_sharded_different_dims(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_zeros(
            [8, 16], mesh, [Sharded(axis=0), Sharded(axis=1)]
        )
        assert dt.shape == [8, 16]
        for shard in dt.local_shards:
            assert list(shard.shape) == [4, 4]


class TestDTensorFactoryValidation:
    def test_uneven_sharding_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="not evenly divisible"):
            DTensor.distributed_ones([7, 8], mesh, [Sharded(axis=0)])

    def test_invalid_shard_dimension_raises(self) -> None:
        mesh = mesh_1d(2)
        with pytest.raises(ValueError, match="out of range"):
            DTensor.distributed_ones([4, 8], mesh, [Sharded(axis=3)])

    def test_negative_shard_dimension_raises(self) -> None:
        mesh = mesh_1d(2)
        with pytest.raises(ValueError, match="out of range"):
            DTensor.distributed_ones([4, 8], mesh, [Sharded(axis=-1)])

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(2)
        with pytest.raises(
            ValueError, match="Need one placement per mesh axis"
        ):
            DTensor.distributed_ones([4, 8], mesh, [Replicated(), Replicated()])

    def test_partial_placement_raises(self) -> None:
        mesh = mesh_1d(2)
        with pytest.raises(ValueError, match="Cannot use Partial placement"):
            DTensor.distributed_ones([4, 8], mesh, [Partial()])


class TestDTensorProperties:
    def test_shape_returns_global_shape(self) -> None:
        mesh = mesh_1d(4)
        dt = DTensor.distributed_ones([12, 8], mesh, [Sharded(axis=0)])
        assert dt.shape == [12, 8]

    def test_dtype_consistent(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 4], mesh, [Replicated()], dtype=DType.float64
        )
        assert dt.dtype == DType.float64

    def test_mesh_returns_mesh(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 4], mesh, [Replicated()])
        assert dt.mesh == mesh

    def test_placements_returns_tuple(self) -> None:
        mesh = mesh_2d(2, 4)
        p: list[Replicated | Sharded | Partial] = [
            Replicated(),
            Sharded(axis=0),
        ]
        dt = DTensor.distributed_ones([8, 16], mesh, p)
        assert dt.placements == tuple(p)
        assert isinstance(dt.placements, tuple)

    def test_local_shards_returns_tuple(self) -> None:
        mesh = mesh_1d(3)
        dt = DTensor.distributed_ones([9, 6], mesh, [Sharded(axis=0)])
        shards = dt.local_shards
        assert isinstance(shards, tuple)
        assert len(shards) == 3
        for shard in shards:
            assert isinstance(shard, Tensor)


class TestDTensorFinalize:
    def test_replicated_returns_first_shard(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        result = dt.finalize()
        assert isinstance(result, Tensor)
        assert list(result.shape) == [4, 8]

    def test_2d_replicated_returns_first_shard(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_ones(
            [4, 8], mesh, [Replicated(), Replicated()]
        )
        result = dt.finalize()
        assert isinstance(result, Tensor)
        assert list(result.shape) == [4, 8]

    def test_sharded_raises(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Sharded(axis=0)])
        with pytest.raises(NotImplementedError, match="non-replicated"):
            dt.finalize()

    def test_partial_raises(self) -> None:
        mesh = mesh_1d(2)
        device = CPU()
        shards = [
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
        ]
        dt = DTensor._from_local_shards(shards, mesh, [Partial()])
        with pytest.raises(NotImplementedError, match="non-replicated"):
            dt.finalize()

    def test_mixed_placement_raises(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated(), Sharded(axis=1)]
        )
        with pytest.raises(NotImplementedError, match="non-replicated"):
            dt.finalize()


class TestDTensorRedistribute:
    def test_always_raises(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        with pytest.raises(NotImplementedError, match="redistribute"):
            dt.redistribute(placements=[Sharded(axis=0)])


class TestDTensorShapeIsDim:
    def test_factory_shape_elements_are_static_dim(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        for d in dt.shape:
            assert isinstance(d, StaticDim)

    def test_factory_shape_equals_int_tuple(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Sharded(axis=0)])
        assert dt.shape == [4, 8]

    def test_from_local_shards_with_symbolic_shape(self) -> None:
        mesh = mesh_1d(2)
        device = CPU()
        shards = [
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
        ]
        sym_shape = (Dim("batch"), Dim(8))
        dt = DTensor._from_local_shards(
            shards, mesh, [Replicated()], shape=sym_shape
        )
        assert isinstance(dt.shape[0], SymbolicDim)
        assert isinstance(dt.shape[1], StaticDim)
        assert dt.shape[0] == "batch"
        assert dt.shape[1] == 8

    def test_from_local_shards_infers_global_shape(self) -> None:
        mesh = mesh_1d(2)
        device = CPU()
        shards = [
            Tensor.full([3, 8], value=1.0, dtype=DType.float32, device=device),
            Tensor.full([3, 8], value=1.0, dtype=DType.float32, device=device),
        ]
        dt = DTensor._from_local_shards(shards, mesh, [Sharded(axis=0)])
        # 3 * 2 = 6 along dim 0
        assert dt.shape == [6, 8]
        for d in dt.shape:
            assert isinstance(d, Dim)

    def test_sharded_symbolic_dim_produces_algebraic_global_shape(self) -> None:
        mesh = mesh_1d(2)
        device = CPU()
        shards = [
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
        ]
        # Provide a symbolic global shape: sharding "seq" along dim 0
        sym_shape = (Dim("seq"), Dim(8))
        dt = DTensor._from_local_shards(
            shards, mesh, [Sharded(axis=0)], shape=sym_shape
        )
        assert isinstance(dt.shape[0], SymbolicDim)
        assert dt.shape[0] == "seq"


class TestDTensorRepr:
    def test_contains_shape(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        r = repr(dt)
        assert "shape=[4, 8]" in r

    def test_contains_dtype(self) -> None:
        mesh = mesh_1d(2)
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        r = repr(dt)
        assert "float32" in r

    def test_contains_mesh(self) -> None:
        mesh = mesh_1d(2, name="tp")
        dt = DTensor.distributed_ones([4, 8], mesh, [Replicated()])
        r = repr(dt)
        assert "DeviceMesh(tp=2)" in r

    def test_contains_placements(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated(), Sharded(axis=1)]
        )
        r = repr(dt)
        assert "Replicated()" in r
        assert "Sharded(axis=1)" in r

    def test_repr_with_symbolic_shape(self) -> None:
        mesh = mesh_1d(2)
        device = CPU()
        shards = [
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
            Tensor.full([4, 8], value=1.0, dtype=DType.float32, device=device),
        ]
        dt = DTensor._from_local_shards(
            shards, mesh, [Replicated()], shape=(Dim("batch"), Dim(8))
        )
        r = repr(dt)
        assert "batch" in r
        assert "8" in r
