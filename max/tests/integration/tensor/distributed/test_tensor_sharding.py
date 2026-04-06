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
"""Tests for Tensor sharding properties and distributed construction.

Covers the sharding-related additions to ``tensor.py``:
``is_distributed``, ``num_shards``, ``mesh``, ``placements``, ``mapping``,
``local_shards``, ``graph_values``, ``buffers``, ``_from_shards``,
``_from_unrealized_shards``, ``_check_not_distributed``, and the
``_storages`` tuple internal representation.

These tests require only CPU — no GPU, no collectives, no dispatch ops.
Module.compile tests live in ``module/test_module_internals.py``.
"""

from __future__ import annotations

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(
        tuple(CPU() for _ in range(rows * cols)), (rows, cols), ("dp", "tp")
    )


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: single-device (unsharded) — backward compat
# ═════════════════════════════════════════════════════════════════════════


class TestSingleDeviceTensor:
    """Ensure unsharded Tensors still behave normally with the new internals."""

    def test_is_not_distributed(self) -> None:
        t = Tensor.zeros([4, 8], dtype=DType.float32, device=CPU())
        assert not t.is_distributed

    def test_num_shards_is_one(self) -> None:
        t = Tensor.ones([3], dtype=DType.float32, device=CPU())
        assert t.num_shards == 1

    def test_local_shards_returns_self(self) -> None:
        t = Tensor.zeros([2, 3], dtype=DType.float32, device=CPU())
        shards = t.local_shards
        assert len(shards) == 1
        assert shards[0] is t

    def test_storage_property(self) -> None:
        t = Tensor.zeros([4], dtype=DType.float32, device=CPU())
        assert t.storage is not None
        assert t.storage.dtype == DType.float32

    def test_state_property(self) -> None:
        t = Tensor.zeros([4], dtype=DType.float32, device=CPU())
        assert t.state is None  # realized

    def test_buffers_returns_single_tuple(self) -> None:
        t = Tensor.ones([2, 3], dtype=DType.float32, device=CPU())
        bufs = t.buffers
        assert isinstance(bufs, tuple)
        assert len(bufs) == 1

    def test_mapping_is_replicated(self) -> None:
        t = Tensor.zeros([4], dtype=DType.float32, device=CPU())
        assert t.mapping.is_fully_replicated

    def test_mesh_is_single(self) -> None:
        t = Tensor.zeros([4], dtype=DType.float32, device=CPU())
        assert t.mesh.is_single
        assert t.mesh.num_devices == 1


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: _from_shards (realized sharded tensor)
# ═════════════════════════════════════════════════════════════════════════


def _make_realized_sharded(
    shape_per_shard: list[int],
    num_shards: int,
    shard_axis: int = 0,
) -> Tensor:
    """Creates a realized sharded Tensor from CPU buffers."""

    mesh = mesh_1d(num_shards)
    bufs = tuple(
        Buffer.zeros(shape_per_shard, dtype=DType.float32, device=CPU())
        for _ in range(num_shards)
    )
    return Tensor._from_shards(
        bufs,
        mesh,
        (Sharded(shard_axis),),
        global_shape=[
            s * num_shards if i == shard_axis else s
            for i, s in enumerate(shape_per_shard)
        ],
    )


class TestFromShards:
    def test_basic_construction(self) -> None:
        t = _make_realized_sharded([2, 8], 4, shard_axis=0)
        assert t.is_distributed
        assert t.num_shards == 4
        assert list(t.shape) == [8, 8]

    def test_shard_axis_1(self) -> None:
        t = _make_realized_sharded([4, 3], 2, shard_axis=1)
        assert t.num_shards == 2
        assert list(t.shape) == [4, 6]

    def test_mesh_and_placements(self) -> None:
        t = _make_realized_sharded([2, 8], 4)
        assert t.mesh.num_devices == 4
        assert t.placements == (Sharded(0),)

    def test_dtype(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        assert t.dtype == DType.float32

    def test_real_is_true(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        assert t.real

    def test_buffers_returns_all_shards(self) -> None:
        t = _make_realized_sharded([2, 4], 4)
        bufs = t.buffers
        assert len(bufs) == 4
        for buf in bufs:
            assert list(buf.shape) == [2, 4]

    def test_local_shards_are_unsharded_tensors(self) -> None:
        t = _make_realized_sharded([3, 4], 2)
        shards = t.local_shards
        assert len(shards) == 2
        for shard in shards:
            assert isinstance(shard, Tensor)
            assert not shard.is_distributed
            assert list(shard.shape) == [3, 4]

    def test_device_count_mismatch_raises(self) -> None:
        mesh = mesh_1d(4)
        bufs = tuple(
            Buffer.zeros([2], dtype=DType.float32, device=CPU())
            for _ in range(2)  # wrong: 2 buffers for 4-device mesh
        )
        with pytest.raises(ValueError, match="Expected 4 storages"):
            Tensor._from_shards(bufs, mesh, (Sharded(0),), global_shape=[8])

    def test_placement_count_mismatch_raises(self) -> None:
        mesh = mesh_1d(2)
        bufs = tuple(
            Buffer.zeros([4], dtype=DType.float32, device=CPU())
            for _ in range(2)
        )
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            Tensor._from_shards(
                bufs, mesh, (Sharded(0), Replicated()), global_shape=[8]
            )


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: sharded property guards (_check_not_distributed)
# ═════════════════════════════════════════════════════════════════════════


class TestDistributedPropertyGuards:
    """Properties that should raise on sharded tensors."""

    def test_storage_raises(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        with pytest.raises(ValueError, match="sharded tensor"):
            _ = t.storage

    def test_state_raises(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        with pytest.raises(ValueError, match="sharded tensor"):
            _ = t.state

    def test_device_raises(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        with pytest.raises(ValueError, match="distributed tensor"):
            _ = t.device

    def test_driver_tensor_raises(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        with pytest.raises(ValueError, match="sharded tensor"):
            _ = t.driver_tensor

    def test_item_raises(self) -> None:
        t = _make_realized_sharded([1], 2)
        with pytest.raises(ValueError, match="sharded tensor"):
            t.item()

    def test_dlpack_raises(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        with pytest.raises(ValueError, match="sharded tensor"):
            t.__dlpack__()


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: shape inference for sharded tensors
# ═════════════════════════════════════════════════════════════════════════


class TestShardedTensorShape:
    def test_global_shape_from_shards(self) -> None:
        t = _make_realized_sharded([3, 8], 4, shard_axis=0)
        assert list(t.shape) == [12, 8]

    def test_explicit_global_shape(self) -> None:
        mesh = mesh_1d(2)
        bufs = tuple(
            Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
            for _ in range(2)
        )
        t = Tensor._from_shards(bufs, mesh, (Sharded(0),), global_shape=[8, 8])
        assert list(t.shape) == [8, 8]

    def test_replicated_shape_equals_shard_shape(self) -> None:
        mesh = mesh_1d(2)
        bufs = tuple(
            Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
            for _ in range(2)
        )
        t = Tensor._from_shards(
            bufs, mesh, (Replicated(),), global_shape=[4, 8]
        )
        assert list(t.shape) == [4, 8]


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: repr for sharded tensors
# ═════════════════════════════════════════════════════════════════════════


class TestShardedTensorRepr:
    def test_contains_shape(self) -> None:
        t = _make_realized_sharded([2, 4], 2, shard_axis=0)
        r = repr(t)
        assert "shape=" in r

    def test_contains_dtype(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        r = repr(t)
        assert "float32" in r

    def test_contains_mapping(self) -> None:
        t = _make_realized_sharded([2, 4], 2)
        r = repr(t)
        assert "mapping=" in r or "PlacementMapping" in r


# ═════════════════════════════════════════════════════════════════════════
#  Tensor: 2D mesh sharding
# ═════════════════════════════════════════════════════════════════════════


class TestTensor2DMesh:
    def test_from_shards_2d(self) -> None:
        mesh = mesh_2d(2, 2)  # 4 devices
        bufs = tuple(
            Buffer.zeros([4, 3], dtype=DType.float32, device=CPU())
            for _ in range(4)
        )
        t = Tensor._from_shards(
            bufs,
            mesh,
            (Sharded(0), Sharded(1)),
            global_shape=[8, 6],
        )
        assert t.is_distributed
        assert t.num_shards == 4
        assert list(t.shape) == [8, 6]
        assert t.placements == (Sharded(0), Sharded(1))

    def test_local_shards_2d(self) -> None:
        mesh = mesh_2d(2, 2)
        bufs = tuple(
            Buffer.zeros([4, 3], dtype=DType.float32, device=CPU())
            for _ in range(4)
        )
        t = Tensor._from_shards(
            bufs, mesh, (Sharded(0), Sharded(1)), global_shape=[8, 6]
        )
        shards = t.local_shards
        assert len(shards) == 4
        for s in shards:
            assert not s.is_distributed
            assert list(s.shape) == [4, 3]
