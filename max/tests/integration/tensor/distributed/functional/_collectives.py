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
"""Shared test logic for collective ops.

DO NOT run this file directly — it contains base classes that are
subclassed by test_collectives_simulated.py and test_collectives_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    partial_fn: Callable   — make_partial (CPU) or gpu_partial (GPU)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from _test_helpers import from_np, shard, to_np
from max.experimental.distributed_functional.collectives import (
    all_gather,
    all_reduce_sum,
    materialize,
    reduce_scatter,
    resolve_partials,
    to_numpy,
)
from max.experimental.distributed_functional.collectives import (
    shard as distribute,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor


class CollectivesTests:
    """Base class with collective test methods.

    Subclass must set MESH_1D, MESH_2D, and partial_fn.
    """

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    # ── AllReduceSum ──────────────────────────────────────────────────

    def test_allreduce_1d(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = all_reduce_sum(t)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), a * 4, rtol=1e-5
        )

    def test_allreduce_2d_tp_axis(self) -> None:
        a = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        t = self.partial_fn(a, self.MESH_2D, (Replicated(), Partial()))
        result = all_reduce_sum(t, mesh_axis="tp")
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                to_np(result.local_shards[i]), a * 2, rtol=1e-5
            )

    def test_allreduce_2d_dp_axis(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Replicated()))
        result = all_reduce_sum(t, mesh_axis="dp")
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                to_np(result.local_shards[i]), a * 2, rtol=1e-5
            )

    # ── AllGather ─────────────────────────────────────────────────────

    def test_allgather_1d_axis0(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        result = all_gather(sharded, tensor_axis=0)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), t_np, rtol=1e-5
        )

    def test_allgather_1d_axis1(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(2, 16)
        sharded = shard(from_np(t_np), self.MESH_1D, [Sharded(1)])
        result = all_gather(sharded, tensor_axis=1)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), t_np, rtol=1e-5
        )

    def test_allgather_2d_tp_axis(self) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        sharded = shard(from_np(t_np), self.MESH_2D, [Replicated(), Sharded(0)])
        result = all_gather(sharded, tensor_axis=0, mesh_axis="tp")
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                to_np(result.local_shards[i]), t_np, rtol=1e-5
            )

    # ── ReduceScatter ─────────────────────────────────────────────────

    def test_reduce_scatter_1d(self) -> None:
        a = np.ones((8, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = reduce_scatter(t, scatter_axis=0)
        assert result.placements == (Sharded(0),)
        for i in range(4):
            np.testing.assert_allclose(
                to_np(result.local_shards[i]),
                np.full((2, 4), 4.0, dtype=np.float32),
                rtol=1e-5,
            )

    def test_reduce_scatter_roundtrip(self) -> None:
        a = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        scattered = reduce_scatter(t, scatter_axis=0)
        assert scattered.placements == (Sharded(0),)
        gathered = all_gather(scattered, tensor_axis=0)
        assert gathered.placements == (Replicated(),)
        np.testing.assert_allclose(
            to_np(gathered.local_shards[0]), a * 4, rtol=1e-5
        )

    def test_reduce_scatter_2d_tp(self) -> None:
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self.partial_fn(a, self.MESH_2D, (Replicated(), Partial()))
        result = reduce_scatter(t, scatter_axis=0, mesh_axis="tp")
        assert result.placements == (Replicated(), Sharded(0))
        expected_full = a * 2
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), expected_full[:2], rtol=1e-5
        )

    # ── ResolvePartials ───────────────────────────────────────────────

    def test_resolve_partial_to_replicated(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = resolve_partials(t)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), a * 4, rtol=1e-5
        )

    def test_resolve_both_axes_partial(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        result = resolve_partials(t)
        assert result.placements == (Replicated(), Replicated())
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), a * 4, rtol=1e-5
        )

    def test_resolve_noop_on_replicated(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = shard(from_np(a), self.MESH_1D, [Replicated()])
        result = resolve_partials(t)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(to_np(result.local_shards[0]), a, rtol=1e-5)

    def test_resolve_preserves_sharded(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Sharded(0), Partial()))
        result = resolve_partials(t)
        assert result.placements[0] == Sharded(0)
        assert result.placements[1] == Replicated()

    # ── Distribute ────────────────────────────────────────────────────

    def test_distribute_sharded_1d(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        mapping = PlacementMapping(self.MESH_1D, (Sharded(0),))
        result = distribute(from_np(t_np), mapping)
        assert result.placements == (Sharded(0),)
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), t_np[:2], rtol=1e-5
        )
        np.testing.assert_allclose(
            to_np(result.local_shards[3]), t_np[6:], rtol=1e-5
        )

    def test_distribute_replicated_1d(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        mapping = PlacementMapping(self.MESH_1D, (Replicated(),))
        result = distribute(from_np(t_np), mapping)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                to_np(result.local_shards[i]), t_np, rtol=1e-5
            )

    def test_distribute_2d_mesh(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        result = distribute(from_np(t_np), mapping)
        assert result.placements == (Replicated(), Sharded(1))
        np.testing.assert_allclose(
            to_np(result.local_shards[0]), t_np[:, :4], rtol=1e-5
        )

    def test_distribute_roundtrip(self) -> None:
        original = np.arange(32, dtype=np.float32).reshape(8, 4)
        distributed = distribute(
            from_np(original), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        gathered = all_gather(distributed, tensor_axis=0)
        np.testing.assert_allclose(
            to_np(gathered.local_shards[0]), original, rtol=1e-5
        )

    def test_distribute_rejects_partial(self) -> None:
        mapping = PlacementMapping(self.MESH_1D, (Partial(),))
        with pytest.raises(ValueError, match="Partial"):
            distribute(from_np(np.ones((2, 4), dtype=np.float32)), mapping)

    # ── Materialize ───────────────────────────────────────────────────

    def test_materialize_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        result = materialize(sharded)
        assert not result.is_distributed
        np.testing.assert_allclose(to_np(result), t_np, rtol=1e-5)

    def test_materialize_partial(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = materialize(t)
        assert not result.is_distributed
        np.testing.assert_allclose(to_np(result), a * 4, rtol=1e-5)

    def test_materialize_noop(self) -> None:
        t = from_np(np.ones((2, 4), dtype=np.float32))
        result = materialize(t)
        assert not result.is_distributed
        np.testing.assert_allclose(
            np.from_dlpack(result), np.ones((2, 4)), rtol=1e-5
        )

    # ── ToNumpy ───────────────────────────────────────────────────────

    def test_to_numpy_distributed(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        result = to_numpy(sharded)
        np.testing.assert_allclose(result, t_np, rtol=1e-5)

    def test_to_numpy_non_distributed(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        result = to_numpy(from_np(t_np))
        np.testing.assert_allclose(result, t_np, rtol=1e-5)
