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
"""Shared test logic for shape-manipulation ops.

DO NOT run this file directly — subclassed by test_shape_*.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    MESH_2:  DeviceMesh   — 2 devices, shape (2,), axis_names=("tp",)
    partial_fn: Callable   — make_partial (CPU) or gpu_partial (GPU)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from _test_helpers import from_np, shard, to_np
from max.dtype import DType
from max.experimental.distributed_functional.shape import (
    concat,
    gather,
    permute,
    reshape,
    split,
    squeeze,
    transpose,
    unsqueeze,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

_F32 = DType.float32


# ── Transpose ────────────────────────────────────────────────────────


class _Transpose:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_s0_swaps_s1(self) -> None:
        """transpose(0,1) with Sharded(0) -> Sharded(1), values correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        out = transpose(t, 0, 1)
        assert out.placements == (Sharded(1),)
        assert list(out.shape) == [8, 4]
        np.testing.assert_allclose(to_np(out), t_np.T, rtol=1e-5)

    def test_unaffected_axis(self) -> None:
        """Transpose axes that don't touch sharded dim — placement unchanged."""
        t_np = np.arange(96, dtype=np.float32).reshape(4, 8, 3)
        t = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        out = transpose(t, 1, 2)
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [4, 3, 8]
        np.testing.assert_allclose(
            to_np(out), t_np.transpose(0, 2, 1), rtol=1e-5
        )


# ── Reshape ──────────────────────────────────────────────────────────


class _Reshape:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_preserved_axis(self) -> None:
        """Reshape keeps sharded dim size -> placement preserved, values correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        out = reshape(t, (8, 2, 2))
        assert out.placements == (Sharded(0),)
        np.testing.assert_allclose(to_np(out), t_np.reshape(8, 2, 2), rtol=1e-5)

    def test_moved_axis(self) -> None:
        """Reshape moves sharded dim to new position."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = shard(from_np(t_np), self.MESH_1D, [Sharded(1)])
        out = reshape(t, (8, 4))
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [8, 4]
        # Each shard: (4, 2) reshaped to (2, 4) — verify per-shard
        for s in out.local_shards:
            arr = to_np(s)
            assert arr.shape == (2, 4)

    def test_partial_passthrough(self) -> None:
        """Partial placement preserved through reshape."""
        data = np.ones((4, 8), dtype=np.float32)
        dt = self.partial_fn(data, self.MESH_1D, (Partial(),))
        out = reshape(dt, (32,))
        assert any(isinstance(p, Partial) for p in out.placements)

    def test_ambiguous_raises(self) -> None:
        """Multiple target dims match sharded size -> ValueError."""
        dt = shard(
            from_np(np.ones((4, 8, 4), dtype=np.float32)),
            self.MESH_1D,
            [Sharded(0)],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            reshape(dt, (4, 4, 8))

    def test_sharded_split_raises(self) -> None:
        """Sharded dim split into multiple dims -> ValueError."""
        dt = shard(
            from_np(np.ones((8, 4), dtype=np.float32)),
            self.MESH_1D,
            [Sharded(0)],
        )
        with pytest.raises(ValueError, match="sharded dim"):
            reshape(dt, (2, 4, 4))


# ── Permute ──────────────────────────────────────────────────────────


class _Permute:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_axis_remapped(self) -> None:
        """Permute remaps sharded axis, values match numpy."""
        t_np = np.arange(96, dtype=np.float32).reshape(4, 8, 3)
        t = shard(from_np(t_np), self.MESH_1D, [Sharded(0)])
        out = permute(t, [2, 0, 1])
        assert out.placements == (Sharded(1),)
        assert list(out.shape) == [3, 4, 8]
        np.testing.assert_allclose(
            to_np(out), t_np.transpose(2, 0, 1), rtol=1e-5
        )


# ── Concat ───────────────────────────────────────────────────────────


class _Concat:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_along_sharded(self) -> None:
        """Concat along sharded axis — placement preserved, global shape doubles."""
        a_np = np.ones((4, 2), dtype=np.float32)
        b_np = np.zeros((4, 2), dtype=np.float32)
        a = shard(from_np(a_np), self.MESH_2, [Sharded(0)])
        b = shard(from_np(b_np), self.MESH_2, [Sharded(0)])
        result = concat((a, b), axis=0)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [8, 2]
        # Each local shard is concat of local shards: [ones(2,2), zeros(2,2)]
        shard_np = to_np(result.local_shards[0])
        assert shard_np.shape == (4, 2)  # 2+2 along axis 0

    def test_along_non_sharded(self) -> None:
        """Concat along non-sharded axis — placement unchanged, values correct."""
        a_np = np.ones((2, 4), dtype=np.float32)
        b_np = np.zeros((2, 4), dtype=np.float32)
        a = shard(from_np(a_np), self.MESH_2, [Sharded(1)])
        b = shard(from_np(b_np), self.MESH_2, [Sharded(1)])
        result = concat((a, b), axis=0)
        assert result.placements == (Sharded(1),)
        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_allclose(to_np(result), expected, rtol=1e-5)


# ── Split ────────────────────────────────────────────────────────────


class _Split:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    @staticmethod
    def _split(t: Tensor, sizes: list[int], axis: int) -> list[Tensor]:
        result = split(t, sizes, axis=axis)
        return [r for r in result if isinstance(r, Tensor)]

    def test_non_sharded_axis(self) -> None:
        """Split along non-sharded axis — each chunk preserves placement + values."""
        t_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        t = shard(from_np(t_np), self.MESH_2, [Sharded(0)])
        parts = self._split(t, [3, 3], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Sharded(0),)
        # Verify values of first chunk
        np.testing.assert_allclose(to_np(parts[0]), t_np[:, :3], rtol=1e-5)
        np.testing.assert_allclose(to_np(parts[1]), t_np[:, 3:], rtol=1e-5)

    def test_along_sharded(self) -> None:
        """Split along sharded axis — per-device shard sizes correct."""
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = shard(from_np(t_np), self.MESH_2, [Sharded(1)])
        parts = self._split(t, [2, 2], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Sharded(1),)

    def test_2d_split_along_sharded(self) -> None:
        """2D mesh split along sharded axis — both mesh-axis placements correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = shard(from_np(t_np), self.MESH_2D, [Replicated(), Sharded(1)])
        parts = self._split(t, [4, 4], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Replicated(), Sharded(1))
            # Each shard's local dim-1 should be 2 (= 4 / tp_size=2)
            assert list(p.local_shards[0].shape)[1] == 2

    def test_uneven_split_raises(self) -> None:
        """Split size not divisible by shard count along sharded axis -> error."""
        t = shard(
            from_np(np.arange(8, dtype=np.float32).reshape(4, 2)),
            self.MESH_2,
            [Sharded(0)],
        )
        # split_sizes=[3, 1] along axis 0 — 3 is not divisible by 2 devices
        with pytest.raises(ValueError, match="not evenly divisible"):
            split(t, [3, 1], axis=0)


# ── Unsqueeze ────────────────────────────────────────────────────────


class _Unsqueeze:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_shifts_sharded(self) -> None:
        """Unsqueeze before sharded axis shifts it up, values preserved."""
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = shard(from_np(t_np), self.MESH_2, [Sharded(0)])
        result = unsqueeze(t, 0)
        assert result.placements == (Sharded(1),)
        assert list(result.shape) == [1, 4, 2]
        np.testing.assert_allclose(
            to_np(result), t_np.reshape(1, 4, 2), rtol=1e-5
        )


# ── Squeeze ──────────────────────────────────────────────────────────


class _Squeeze:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_shifts_sharded(self) -> None:
        """Squeeze before sharded axis shifts it down, values preserved."""
        t_np = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
        t = shard(from_np(t_np), self.MESH_2, [Sharded(1)])
        result = squeeze(t, 0)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 2]
        np.testing.assert_allclose(to_np(result), t_np.reshape(4, 2), rtol=1e-5)

    def test_sharded_axis_raises(self) -> None:
        """Squeezing the sharded axis itself raises."""
        t_np = np.ones((2, 1, 3), dtype=np.float32)
        t = shard(from_np(t_np), self.MESH_2, [Sharded(1)])
        with pytest.raises(ValueError, match="sharded axis"):
            squeeze(t, 1)


# ── Gather ───────────────────────────────────────────────────────────


class _Gather:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_gather_axis(self) -> None:
        """Gather along non-sharded axis — placement preserved, values correct."""
        w_np = np.arange(20, dtype=np.float32).reshape(5, 4)
        weight = shard(from_np(w_np), self.MESH_2, [Sharded(1)])
        indices = shard(
            from_np(np.array([0, 3], dtype=np.int64)),
            self.MESH_2,
            [Replicated()],
        )
        result = gather(weight, indices, axis=0)
        assert result.placements == (Sharded(1),)
        expected = w_np[[0, 3]]
        np.testing.assert_allclose(to_np(result), expected, rtol=1e-5)

    def test_gather_axis_raises(self) -> None:
        """Gathering along sharded axis raises."""
        weight = shard(
            from_np(np.arange(20, dtype=np.float32).reshape(10, 2)),
            self.MESH_2,
            [Sharded(0)],
        )
        indices = shard(
            from_np(np.array([0, 5], dtype=np.int64)),
            self.MESH_2,
            [Replicated()],
        )
        with pytest.raises(ValueError, match="sharded axis"):
            gather(weight, indices, axis=0)


# ── Combined base class ─────────────────────────────────────────────


class ShapeTests(
    _Transpose,
    _Reshape,
    _Permute,
    _Concat,
    _Split,
    _Unsqueeze,
    _Squeeze,
    _Gather,
):
    """Aggregates all shape test classes for thin subclassing."""

    pass
