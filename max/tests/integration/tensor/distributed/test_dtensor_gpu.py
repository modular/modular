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
"""Tests for DTensor on real GPU devices with multi-GPU meshes."""

from __future__ import annotations

import pytest
from max.driver import Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental.distributed import (
    DeviceMesh,
    DTensor,
    Partial,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import StaticDim


def gpu_devices(n: int) -> tuple[Device, ...]:
    """Returns a tuple of n distinct Accelerator devices."""
    count = accelerator_count()
    if count < n:
        pytest.skip(f"Test requires {n} GPUs, found {count}")
    return tuple(Accelerator(i) for i in range(n))


def gpu_mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    """Creates a 1D GPU device mesh with n devices."""
    devices = gpu_devices(n)
    return DeviceMesh(devices=devices, mesh_shape=(n,), axis_names=(name,))


def gpu_mesh_2d(rows: int, cols: int) -> DeviceMesh:
    """Creates a 2D GPU device mesh with rows * cols devices."""
    devices = gpu_devices(rows * cols)
    return DeviceMesh(
        devices=devices, mesh_shape=(rows, cols), axis_names=("dp", "tp")
    )


# -- 2-GPU tests --


requires_2_gpus = pytest.mark.skipif(
    accelerator_count() < 2, reason="Requires at least 2 GPUs"
)


@requires_2_gpus
class TestDTensorOnes2GPU:
    def test_creates_correct_shard_count(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones([8, 16], mesh, [Replicated()])
        assert len(dt.local_shards) == 2

    def test_replicated_shard_shape(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones([8, 16], mesh, [Replicated()])
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 16]

    def test_sharded_shard_shape(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [4, 16]

    def test_global_shape(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=0)])
        assert dt.shape == [8, 16]

    def test_dtype_default(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones([4, 4], mesh, [Replicated()])
        # Accelerator default dtype is bfloat16
        assert dt.dtype == DType.bfloat16

    def test_dtype_explicit(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 4], mesh, [Replicated()], dtype=DType.float32
        )
        assert dt.dtype == DType.float32


@requires_2_gpus
class TestDTensorZeros2GPU:
    def test_creates_correct_shard_count(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_zeros([6, 4], mesh, [Sharded(axis=0)])
        assert len(dt.local_shards) == 2

    def test_shard_shape(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_zeros([6, 4], mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [3, 4]


@requires_2_gpus
class TestDTensorFull2GPU:
    def test_custom_value(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_full([4, 4], 42.0, mesh, [Replicated()])
        assert dt.shape == [4, 4]
        assert len(dt.local_shards) == 2

    def test_sharded_divides_dimension(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_full([12, 8], 3.14, mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [6, 8]
        assert dt.shape == [12, 8]


@requires_2_gpus
class TestDTensorProperties2GPU:
    def test_shape_elements_are_dim(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Sharded(axis=0)], dtype=DType.float32
        )
        for d in dt.shape:
            assert isinstance(d, StaticDim)
        assert dt.shape == [8, 16]

    def test_shards_on_distinct_devices(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated()], dtype=DType.float32
        )
        devices = [shard.device for shard in dt.local_shards]
        assert devices[0] != devices[1]

    def test_device_raises_for_multi_shard(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated()], dtype=DType.float32
        )
        with pytest.raises(ValueError, match="does not have a single device"):
            _ = dt.device

    def test_mesh_returns_mesh(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 4], mesh, [Replicated()], dtype=DType.float32
        )
        assert dt.mesh == mesh

    def test_placements_returns_tuple(self) -> None:
        mesh = gpu_mesh_1d(2)
        placements = [Sharded(axis=1)]
        dt = DTensor.distributed_ones(
            [8, 16], mesh, placements, dtype=DType.float32
        )
        assert dt.placements == tuple(placements)

    def test_all_shards_realized(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 4], mesh, [Replicated()], dtype=DType.float32
        )
        assert dt.real


@requires_2_gpus
class TestDTensorFinalize2GPU:
    def test_replicated_returns_first_shard(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 8], mesh, [Replicated()], dtype=DType.float32
        )
        result = dt.finalize()
        assert isinstance(result, Tensor)
        assert list(result.shape) == [4, 8]

    def test_sharded_raises(self) -> None:
        mesh = gpu_mesh_1d(2)
        dt = DTensor.distributed_ones(
            [4, 8], mesh, [Sharded(axis=0)], dtype=DType.float32
        )
        with pytest.raises(NotImplementedError, match="non-replicated"):
            dt.finalize()


@requires_2_gpus
class TestDTensorValidation2GPU:
    def test_uneven_sharding_raises(self) -> None:
        mesh = gpu_mesh_1d(2)
        with pytest.raises(ValueError, match="not evenly divisible"):
            DTensor.distributed_ones([7, 8], mesh, [Sharded(axis=0)])

    def test_partial_placement_raises(self) -> None:
        mesh = gpu_mesh_1d(2)
        with pytest.raises(ValueError, match="Cannot use Partial placement"):
            DTensor.distributed_ones([4, 8], mesh, [Partial()])


# -- 4-GPU tests --


requires_4_gpus = pytest.mark.skipif(
    accelerator_count() < 4, reason="Requires at least 4 GPUs"
)


@requires_4_gpus
class TestDTensorOnes4GPU:
    def test_creates_correct_shard_count(self) -> None:
        mesh = gpu_mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Replicated()])
        assert len(dt.local_shards) == 4

    def test_sharded_shard_shape(self) -> None:
        mesh = gpu_mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=0)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [2, 16]

    def test_sharded_dim1(self) -> None:
        mesh = gpu_mesh_1d(4)
        dt = DTensor.distributed_ones([8, 16], mesh, [Sharded(axis=1)])
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 4]
        assert dt.shape == [8, 16]


@requires_4_gpus
class TestDTensorFactory2DMesh4GPU:
    def test_replicated_replicated(self) -> None:
        mesh = gpu_mesh_2d(2, 2)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated(), Replicated()], dtype=DType.float32
        )
        assert dt.shape == [8, 16]
        assert len(dt.local_shards) == 4
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 16]

    def test_replicated_sharded(self) -> None:
        mesh = gpu_mesh_2d(2, 2)
        dt = DTensor.distributed_ones(
            [8, 16],
            mesh,
            [Replicated(), Sharded(axis=1)],
            dtype=DType.float32,
        )
        assert dt.shape == [8, 16]
        for shard in dt.local_shards:
            assert list(shard.shape) == [8, 8]

    def test_sharded_sharded_different_dims(self) -> None:
        mesh = gpu_mesh_2d(2, 2)
        dt = DTensor.distributed_zeros(
            [8, 16],
            mesh,
            [Sharded(axis=0), Sharded(axis=1)],
            dtype=DType.float32,
        )
        assert dt.shape == [8, 16]
        for shard in dt.local_shards:
            assert list(shard.shape) == [4, 8]


@requires_4_gpus
class TestDTensorProperties4GPU:
    def test_shards_on_distinct_devices(self) -> None:
        mesh = gpu_mesh_1d(4)
        dt = DTensor.distributed_ones(
            [8, 16], mesh, [Replicated()], dtype=DType.float32
        )
        devices = [shard.device for shard in dt.local_shards]
        assert len(set(str(d) for d in devices)) == 4

    def test_2d_mesh_shard_count(self) -> None:
        mesh = gpu_mesh_2d(2, 2)
        dt = DTensor.distributed_ones(
            [8, 16],
            mesh,
            [Sharded(axis=0), Sharded(axis=1)],
            dtype=DType.float32,
        )
        assert len(dt.local_shards) == 4

    def test_finalize_replicated_2d(self) -> None:
        mesh = gpu_mesh_2d(2, 2)
        dt = DTensor.distributed_ones(
            [4, 8],
            mesh,
            [Replicated(), Replicated()],
            dtype=DType.float32,
        )
        result = dt.finalize()
        assert isinstance(result, Tensor)
        assert list(result.shape) == [4, 8]
