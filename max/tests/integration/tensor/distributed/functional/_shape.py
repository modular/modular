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
    partial_fn: Callable   — make_partial (CPU) or make_partial (GPU)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.functional import (
    argsort,
    bottom_k,
    broadcast_to,
    buffer_store,
    buffer_store_slice,
    chunk,
    concat,
    flatten,
    gather,
    gather_nd,
    outer,
    pad,
    permute,
    repeat_interleave,
    reshape,
    scatter,
    scatter_add,
    scatter_nd,
    scatter_nd_add,
    slice_tensor,
    split,
    squeeze,
    stack,
    tile,
    top_k,
    transfer_to,
    transpose,
    unsqueeze,
    where,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
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
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        out = transpose(t, 0, 1)
        assert out.placements == (Sharded(1),)
        assert list(out.shape) == [8, 4]
        np.testing.assert_allclose(out.to_numpy(), t_np.T, rtol=1e-5)

    def test_unaffected_axis(self) -> None:
        """Transpose axes that don't touch sharded dim — placement unchanged."""
        t_np = np.arange(96, dtype=np.float32).reshape(4, 8, 3)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        out = transpose(t, 1, 2)
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [4, 3, 8]
        np.testing.assert_allclose(
            out.to_numpy(), t_np.transpose(0, 2, 1), rtol=1e-5
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
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        out = reshape(t, (8, 2, 2))
        assert out.placements == (Sharded(0),)
        np.testing.assert_allclose(
            out.to_numpy(), t_np.reshape(8, 2, 2), rtol=1e-5
        )

    def test_merged_axis_raises(self) -> None:
        """Reshape that merges sharded axis with adjacent axis raises.

        (4, 8) Sharded(1) → (8, 4) merges axis 0 and axis 1 into new
        axis 0. Per-shard reshape would give wrong global ordering.
        The cumulative-product factorization correctly detects this.
        """
        t = transfer_to(
            Tensor(np.arange(32, dtype=np.float32).reshape(4, 8)),
            PlacementMapping(self.MESH_1D, (Sharded(1),)),
        )
        with pytest.raises(ValueError, match="cannot be mapped"):
            reshape(t, (8, 4))

    def test_partial_passthrough(self) -> None:
        """Partial placement preserved through reshape."""
        data = np.ones((4, 8), dtype=np.float32)
        dt = self.partial_fn(data, self.MESH_1D, (Partial(),))
        out = reshape(dt, (32,))
        assert any(isinstance(p, Partial) for p in out.placements)

    def test_ambiguous_now_resolved(self) -> None:
        """Ambiguous size match resolved by cumulative-product factorization.

        Old heuristic: (4, 8, 4) Sharded(0) → (4, 4, 8) has two dims of
        size 4, raised "ambiguous". New approach: cumulative products show
        old axis 0 maps to new axis 0 (both start at boundary 0, end at 4).
        """
        t_np = np.arange(128, dtype=np.float32).reshape(4, 8, 4)
        dt = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        out = reshape(dt, (4, 4, 8))
        assert out.placements == (Sharded(0),)
        np.testing.assert_allclose(
            out.to_numpy(), t_np.reshape(4, 4, 8), rtol=1e-5
        )

    def test_sharded_split_raises(self) -> None:
        """Sharded dim split with indivisible shard size -> ValueError."""
        dt = transfer_to(
            Tensor(np.ones((8, 4), dtype=np.float32)),
            PlacementMapping(self.MESH_1D, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="not evenly divisible"):
            reshape(dt, (2, 4, 4))

    def test_mha_reshape_batch_sharded(self) -> None:
        """Multi-head attention reshape: [B, S, H*D] → [B, S, H, D].

        Sharded on batch (axis 0) — preserved through reshape since
        the batch axis maps 1:1 via cumulative products.
        """
        B, S, H, D = 4, 8, 4, 16
        t_np = np.arange(B * S * H * D, dtype=np.float32).reshape(B, S, H * D)
        dt = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        out = reshape(dt, (B, S, H, D))
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [B, S, H, D]
        np.testing.assert_allclose(
            out.to_numpy(), t_np.reshape(B, S, H, D), rtol=1e-5
        )


# ── Permute ──────────────────────────────────────────────────────────


class _Permute:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_axis_remapped(self) -> None:
        """Permute remaps sharded axis, values match numpy."""
        t_np = np.arange(96, dtype=np.float32).reshape(4, 8, 3)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        out = permute(t, [2, 0, 1])
        assert out.placements == (Sharded(1),)
        assert list(out.shape) == [3, 4, 8]
        np.testing.assert_allclose(
            out.to_numpy(), t_np.transpose(2, 0, 1), rtol=1e-5
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
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = concat((a, b), axis=0)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [8, 2]
        # Each local shard is concat of local shards: [ones(2,2), zeros(2,2)]
        shard_np = result.local_shards[0].to_numpy()
        assert shard_np.shape == (4, 2)  # 2+2 along axis 0

    def test_along_non_sharded(self) -> None:
        """Concat along non-sharded axis — placement unchanged, values correct."""
        a_np = np.ones((2, 4), dtype=np.float32)
        b_np = np.zeros((2, 4), dtype=np.float32)
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        result = concat((a, b), axis=0)
        assert result.placements == (Sharded(1),)
        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


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
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        parts = self._split(t, [3, 3], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Sharded(0),)
        # Verify values of first chunk
        np.testing.assert_allclose(parts[0].to_numpy(), t_np[:, :3], rtol=1e-5)
        np.testing.assert_allclose(parts[1].to_numpy(), t_np[:, 3:], rtol=1e-5)

    def test_along_sharded(self) -> None:
        """Split along sharded axis — per-device shard sizes correct."""
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        parts = self._split(t, [2, 2], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Sharded(1),)

    def test_2d_split_along_sharded(self) -> None:
        """2D mesh split along sharded axis — both mesh-axis placements correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        parts = self._split(t, [4, 4], axis=1)
        assert len(parts) == 2
        for p in parts:
            assert p.placements == (Replicated(), Sharded(1))
            # Each shard's local dim-1 should be 2 (= 4 / tp_size=2)
            assert list(p.local_shards[0].shape)[1] == 2

    def test_uneven_split_raises(self) -> None:
        """Split size not divisible by shard count along sharded axis -> error."""
        t = transfer_to(
            Tensor(np.arange(8, dtype=np.float32).reshape(4, 2)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
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
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = unsqueeze(t, 0)
        assert result.placements == (Sharded(1),)
        assert list(result.shape) == [1, 4, 2]
        np.testing.assert_allclose(
            result.to_numpy(), t_np.reshape(1, 4, 2), rtol=1e-5
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
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        result = squeeze(t, 0)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 2]
        np.testing.assert_allclose(
            result.to_numpy(), t_np.reshape(4, 2), rtol=1e-5
        )

    def test_sharded_axis_raises(self) -> None:
        """Squeezing the sharded axis itself raises."""
        t_np = np.ones((2, 1, 3), dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
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
        weight = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        indices = transfer_to(
            Tensor(np.array([0, 3], dtype=np.int64)),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        result = gather(weight, indices, axis=0)
        assert result.placements == (Sharded(1),)
        expected = w_np[[0, 3]]
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_gather_axis_raises(self) -> None:
        """Gathering along sharded axis raises."""
        weight = transfer_to(
            Tensor(np.arange(20, dtype=np.float32).reshape(10, 2)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        indices = transfer_to(
            Tensor(np.array([0, 5], dtype=np.int64)),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            gather(weight, indices, axis=0)


# ── BroadcastTo ─────────────────────────────────────────────────────


class _BroadcastTo:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_replicated_broadcast(self) -> None:
        """Broadcast a replicated tensor — placement preserved, values correct."""
        t_np = np.ones((1, 4), dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = broadcast_to(t, shape=(3, 4))
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [3, 4]
        np.testing.assert_allclose(
            result.to_numpy(), np.broadcast_to(t_np, (3, 4)), rtol=1e-5
        )

    def test_sharded_non_broadcast_dim(self) -> None:
        """Broadcast doesn't touch sharded dim — placement preserved."""
        t_np = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = broadcast_to(t, shape=(4, 3, 2))
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 3, 2]
        np.testing.assert_allclose(
            result.to_numpy(), np.broadcast_to(t_np, (4, 3, 2)), rtol=1e-5
        )

    def test_partial_passthrough(self) -> None:
        """Partial placement preserved through broadcast."""
        data = np.ones((1, 4), dtype=np.float32)
        dt = self.partial_fn(data, self.MESH_1D, (Partial(),))
        result = broadcast_to(dt, shape=(3, 4))
        assert any(isinstance(p, Partial) for p in result.placements)
        assert list(result.shape) == [3, 4]


# ── Flatten ─────────────────────────────────────────────────────────


class _Flatten:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_replicated_flatten(self) -> None:
        """Flatten a replicated tensor — placement preserved, values correct."""
        t_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = flatten(t, start_dim=1, end_dim=2)
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [2, 12]
        np.testing.assert_allclose(
            result.to_numpy(), t_np.reshape(2, 12), rtol=1e-5
        )

    def test_sharded_outside_range(self) -> None:
        """Sharded axis outside flatten range — shifts down correctly."""
        t_np = np.arange(48, dtype=np.float32).reshape(4, 3, 2, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = flatten(t, start_dim=1, end_dim=2)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 6, 2]
        np.testing.assert_allclose(
            result.to_numpy(), t_np.reshape(4, 6, 2), rtol=1e-5
        )

    def test_sharded_in_range_raises(self) -> None:
        """Flattening across sharded axis raises."""
        t_np = np.ones((4, 3, 2), dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(1),))
        )
        with pytest.raises(ValueError, match="sharded axis"):
            flatten(t, start_dim=0, end_dim=1)

    def test_partial_passthrough(self) -> None:
        """Partial placement preserved through flatten."""
        data = np.ones((2, 3, 4), dtype=np.float32)
        dt = self.partial_fn(data, self.MESH_1D, (Partial(),))
        result = flatten(dt, start_dim=0, end_dim=-1)
        assert any(isinstance(p, Partial) for p in result.placements)


# ── Combined base class ─────────────────────────────────────────────

# ── Stack ───────────────────────────────────────────────────────────


class _Stack:
    MESH_2: ClassVar[DeviceMesh]

    def test_stack_replicated(self) -> None:
        """Stack two replicated tensors along a new axis."""
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.zeros((2, 3), dtype=np.float32)
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = stack((a, b), axis=0)
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [2, 2, 3]
        np.testing.assert_allclose(
            result.to_numpy(), np.stack([a_np, b_np], axis=0), rtol=1e-5
        )

    def test_stack_sharded_shifts(self) -> None:
        """Stacking before sharded axis shifts it up."""
        a_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        b_np = np.arange(8, 16, dtype=np.float32).reshape(4, 2)
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = stack((a, b), axis=0)
        # Sharded(0) shifts to Sharded(1) because new axis inserted at 0
        assert result.placements == (Sharded(1),)
        assert list(result.shape) == [2, 4, 2]


# ── Argsort ─────────────────────────────────────────────────────────


class _Argsort:
    MESH_2: ClassVar[DeviceMesh]

    def test_argsort_replicated(self) -> None:
        """Argsort on replicated 1D tensor — placement preserved."""
        t_np = np.array([3, 1, 2, 0], dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = argsort(t)
        assert result.placements == (Replicated(),)
        expected = np.argsort(t_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)


# ── TopK / BottomK ──────────────────────────────────────────────────


class _TopKBottomK:
    MESH_2: ClassVar[DeviceMesh]

    def test_top_k_non_sharded(self) -> None:
        """top_k along non-sharded axis — placement preserved."""
        t_np = np.array([[5, 1, 3, 2, 4], [10, 8, 6, 9, 7]], dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        values, indices = top_k(t, k=2, axis=1)
        assert values.placements == (Sharded(0),)
        assert indices.placements == (Sharded(0),)

    def test_top_k_sharded_raises(self) -> None:
        """top_k along sharded axis raises."""
        t = transfer_to(
            Tensor(np.ones((4, 2), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            top_k(t, k=1, axis=0)

    def test_bottom_k_non_sharded(self) -> None:
        """bottom_k along non-sharded axis — placement preserved."""
        t_np = np.array([[5, 1, 3, 2, 4], [10, 8, 6, 9, 7]], dtype=np.float32)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        values, indices = bottom_k(t, k=2, axis=1)
        assert values.placements == (Sharded(0),)
        assert indices.placements == (Sharded(0),)


# ── Chunk ───────────────────────────────────────────────────────────


class _Chunk:
    MESH_2: ClassVar[DeviceMesh]

    def test_chunk_non_sharded(self) -> None:
        """Chunk along non-sharded axis — each chunk preserves placement."""
        t_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        parts = chunk(t, chunks=3, axis=1)
        assert len(parts) == 3
        for p in parts:
            assert p.placements == (Sharded(0),)

    def test_chunk_sharded_raises(self) -> None:
        """Chunk along sharded axis raises."""
        t = transfer_to(
            Tensor(np.ones((4, 6), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            chunk(t, chunks=2, axis=0)


# ── RepeatInterleave ────────────────────────────────────────────────


class _RepeatInterleave:
    MESH_2: ClassVar[DeviceMesh]

    def test_repeat_non_sharded(self) -> None:
        """repeat_interleave along non-sharded axis — placement preserved."""

        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("repeat_interleave not supported on GPU")
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = repeat_interleave(t, repeats=3, axis=1)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 6]
        np.testing.assert_allclose(
            result.to_numpy(), np.repeat(t_np, 3, axis=1), rtol=1e-5
        )

    def test_repeat_sharded_raises(self) -> None:
        """repeat_interleave along sharded axis raises."""
        t = transfer_to(
            Tensor(np.ones((4, 2), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            repeat_interleave(t, repeats=2, axis=0)

    def test_repeat_axis_none_sharded_raises(self) -> None:
        """repeat_interleave with axis=None on Sharded input raises.

        axis=None flattens the input first, which produces wrong global
        ordering when the input is sharded (each shard flattens
        independently).
        """
        t = transfer_to(
            Tensor(np.ones((4, 2), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded"):
            repeat_interleave(t, repeats=2, axis=None)


# ── Tile ────────────────────────────────────────────────────────────


class _Tile:
    MESH_2: ClassVar[DeviceMesh]

    def _skip_if_gpu(self) -> None:
        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("Tile Mojo kernel is MO_HostOnly (CPU-only)")

    def test_tile_replicated(self) -> None:
        """Tile a replicated tensor — placement preserved."""
        self._skip_if_gpu()
        t_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = tile(t, [2, 1])
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [4, 3]
        np.testing.assert_allclose(
            result.to_numpy(), np.tile(t_np, (2, 1)), rtol=1e-5
        )

    def test_tile_sharded(self) -> None:
        """Tile on batch-sharded tensor — placement preserved."""
        self._skip_if_gpu()
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = tile(t, [1, 3])
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 6]


# ── Pad ─────────────────────────────────────────────────────────────


class _Pad:
    MESH_2: ClassVar[DeviceMesh]

    def _skip_if_gpu(self) -> None:
        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip(
                "PadConstant Mojo kernel GPU has InlineArray capture bug"
            )

    def test_pad_replicated(self) -> None:
        """Pad a replicated tensor — placement preserved."""
        self._skip_if_gpu()
        t_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        # Flat paddings: [before_dim0, after_dim0, before_dim1, after_dim1]
        result = pad(t, [0, 0, 1, 1])
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [2, 5]

    def test_pad_sharded_non_padded(self) -> None:
        """Pad non-sharded dim on a sharded tensor."""
        self._skip_if_gpu()
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        # Pad dim 1 only
        result = pad(t, [0, 0, 1, 1])
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 4]

    def test_pad_sharded_dim_raises(self) -> None:
        """Padding along sharded axis raises."""
        t = transfer_to(
            Tensor(np.ones((4, 2), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            pad(t, [1, 1, 0, 0])


# ── SliceTensor ─────────────────────────────────────────────────────


class _SliceTensor:
    MESH_2: ClassVar[DeviceMesh]

    def test_slice_replicated(self) -> None:
        """Slice a replicated tensor — placement preserved."""
        t_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = slice_tensor(t, [slice(1, 3)])
        assert result.placements == (Replicated(),)
        assert list(result.shape) == [2, 5]
        np.testing.assert_allclose(result.to_numpy(), t_np[1:3], rtol=1e-5)

    def test_slice_sharded_non_sliced(self) -> None:
        """Slice non-sharded dim on a sharded tensor."""

        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("slice_tensor graph op crashes on GPU")
        t_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = slice_tensor(t, [slice(None), slice(0, 3)])
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 3]
        np.testing.assert_allclose(result.to_numpy(), t_np[:, :3], rtol=1e-5)


# ── Scatter / ScatterAdd ────────────────────────────────────────────


class _Scatter:
    MESH_2: ClassVar[DeviceMesh]

    def test_scatter_non_sharded_axis(self) -> None:
        """Scatter along non-sharded axis — placement preserved.

        ops.scatter(input, updates, indices, axis).
        scatter/scatter_add are CPU-only Mojo kernels.
        """

        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("scatter is CPU-only")

        x_np = np.zeros((4, 4), dtype=np.float32)
        updates_np = np.ones((4, 2), dtype=np.float32)
        idx_np = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        updates = transfer_to(
            Tensor(updates_np),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        idx = transfer_to(
            Tensor(idx_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = scatter(x, updates, idx, axis=1)
        assert result.placements == (Sharded(0),)

    def test_scatter_sharded_axis_raises(self) -> None:
        """Scatter along sharded axis raises."""
        x = transfer_to(
            Tensor(np.zeros((4, 4), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        updates = transfer_to(
            Tensor(np.ones((4, 2), dtype=np.float32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        idx = transfer_to(
            Tensor(np.zeros((4, 2), dtype=np.int32)),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="sharded axis"):
            scatter(x, updates, idx, axis=0)

    def test_scatter_add_non_sharded_axis(self) -> None:
        """scatter_add along non-sharded axis — placement preserved."""

        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("scatter_add is CPU-only")
        x_np = np.zeros((4, 4), dtype=np.float32)
        updates_np = np.ones((4, 2), dtype=np.float32)
        idx_np = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        updates = transfer_to(
            Tensor(updates_np),
            PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        idx = transfer_to(
            Tensor(idx_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = scatter_add(x, updates, idx, axis=1)
        assert result.placements == (Sharded(0),)


# ── Outer ───────────────────────────────────────────────────────────


class _Outer:
    MESH_2: ClassVar[DeviceMesh]

    def test_outer_replicated(self) -> None:
        """Outer product of two replicated vectors."""
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5], dtype=np.float32)
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = outer(a, b)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.to_numpy(), np.outer(a_np, b_np), rtol=1e-5
        )


# ── Where (ternary elementwise) ─────────────────────────────────────


class _Where:
    MESH_2: ClassVar[DeviceMesh]

    def test_where_replicated(self) -> None:
        """Where on replicated tensors — placement preserved."""
        cond_np = np.array([[True, False], [False, True]])
        x_np = np.ones((2, 2), dtype=np.float32)
        y_np = np.zeros((2, 2), dtype=np.float32)
        cond = transfer_to(
            Tensor(cond_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        y = transfer_to(
            Tensor(y_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = where(cond, x, y)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.to_numpy(), np.where(cond_np, x_np, y_np), rtol=1e-5
        )

    def test_where_sharded(self) -> None:
        """Where on batch-sharded tensors — placement preserved."""
        cond_np = np.array([[True, False], [False, True]])
        x_np = np.ones((2, 2), dtype=np.float32)
        y_np = np.zeros((2, 2), dtype=np.float32)
        cond = transfer_to(
            Tensor(cond_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        y = transfer_to(
            Tensor(y_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = where(cond, x, y)
        assert result.placements == (Sharded(0),)
        np.testing.assert_allclose(
            result.to_numpy(), np.where(cond_np, x_np, y_np), rtol=1e-5
        )

    def test_where_replicated_cond_sharded_data(self) -> None:
        """Where with Replicated cond and Sharded x/y picks Sharded output.

        Previously, where_impl used only cond.placements for the output.
        The fix derives output placements from all three inputs, preferring
        non-Replicated. align_shards splits the Replicated cond to match
        the Sharded local sizes.
        """
        cond_np = np.array(
            [[True, False], [False, True], [True, True], [False, False]]
        )
        x_np = np.ones((4, 2), dtype=np.float32) * 10.0
        y_np = np.zeros((4, 2), dtype=np.float32)
        cond = transfer_to(
            Tensor(cond_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        y = transfer_to(
            Tensor(y_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = where(cond, x, y)
        assert result.placements == (Sharded(0),)
        assert tuple(result.shape) == (4, 2)
        np.testing.assert_allclose(
            result.to_numpy(), np.where(cond_np, x_np, y_np), rtol=1e-5
        )


# ── GatherNd ───────────────────────────────────────────────────────


class _GatherNd:
    MESH_2: ClassVar[DeviceMesh]

    def test_gather_nd_replicated(self) -> None:
        """gather_nd on replicated tensor — placement preserved."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([[0], [2]], dtype=np.int32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        idx = transfer_to(
            Tensor(idx_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        result = gather_nd(x, idx)
        assert result.placements == (Replicated(),)
        # gather_nd with indices [[0],[2]] selects rows 0 and 2
        expected = x_np[[0, 2]]
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


# ── ScatterNd / ScatterNdAdd ───────────────────────────────────────


class _ScatterNd:
    MESH_2: ClassVar[DeviceMesh]

    def test_scatter_nd_replicated(self) -> None:
        """scatter_nd on replicated tensor — placement preserved."""
        x_np = np.zeros((4, 4), dtype=np.float32)
        updates_np = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        idx_np = np.array([[0], [2]], dtype=np.int32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        updates = transfer_to(
            Tensor(updates_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        idx = transfer_to(
            Tensor(idx_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        result = scatter_nd(x, updates, idx)
        assert result.placements == (Replicated(),)

    def test_scatter_nd_add_replicated(self) -> None:
        """scatter_nd_add on replicated tensor — placement preserved.

        scatter_nd_add is CPU-only (the Mojo kernel does not support GPU).
        """

        if not isinstance(self.MESH_2.devices[0], CPU):
            pytest.skip("scatter_nd_add is CPU-only")
        x_np = np.ones((4, 4), dtype=np.float32)
        updates_np = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        idx_np = np.array([[0], [2]], dtype=np.int32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        updates = transfer_to(
            Tensor(updates_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        idx = transfer_to(
            Tensor(idx_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        result = scatter_nd_add(x, updates, idx)
        assert result.placements == (Replicated(),)


# ── Buffer store (in-place full update) ──────────────────────────────


class _BufferStore:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]

    def test_replicated_full_store(self) -> None:
        """Replicated tensor: full store replaces all values."""
        dest_np = np.zeros((4, 4), dtype=np.float32)
        src_np = np.ones((4, 4), dtype=np.float32) * 5.0
        dest = transfer_to(
            Tensor(dest_np),
            PlacementMapping(self.MESH_1D, (Replicated(),)),
        )
        src = transfer_to(
            Tensor(src_np),
            PlacementMapping(self.MESH_1D, (Replicated(),)),
        )
        buffer_store(dest, src)
        np.testing.assert_allclose(dest.to_numpy(), src_np)

    def test_sharded_full_store(self) -> None:
        """Sharded(0) tensor: full store replaces per-shard values."""
        dest_np = np.zeros((4, 6), dtype=np.float32)
        src_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        dest = transfer_to(
            Tensor(dest_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        src = transfer_to(
            Tensor(src_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        buffer_store(dest, src)
        np.testing.assert_allclose(dest.to_numpy(), src_np)

    def test_non_distributed_full_store(self) -> None:
        """Non-distributed tensor: full store works directly."""
        dest_np = np.zeros((3, 3), dtype=np.float32)
        src_np = np.ones((3, 3), dtype=np.float32) * 2.0
        dest = Tensor(dest_np)
        src = Tensor(src_np)
        buffer_store(dest, src)
        np.testing.assert_allclose(dest.to_numpy(), src_np)


# ── Slice store (in-place update) ────────────────────────────────────


class _SliceStore:
    MESH_1D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]

    def test_replicated_row_slice(self) -> None:
        """Replicated tensor: row slice update preserves values."""
        dest_np = np.zeros((8, 4), dtype=np.float32)
        src_np = np.ones((2, 4), dtype=np.float32) * 7.0
        dest = transfer_to(
            Tensor(dest_np),
            PlacementMapping(self.MESH_1D, (Replicated(),)),
        )
        src = transfer_to(
            Tensor(src_np),
            PlacementMapping(self.MESH_1D, (Replicated(),)),
        )
        buffer_store_slice(dest, src, (slice(2, 4), slice(None)))
        expected = dest_np.copy()
        expected[2:4, :] = src_np
        np.testing.assert_allclose(dest.to_numpy(), expected)

    def test_sharded_row_slice(self) -> None:
        """Sharded(0) tensor: row slice applied per shard."""
        dest_np = np.zeros((8, 4), dtype=np.float32)
        src_np = np.ones((1, 4), dtype=np.float32) * 3.0
        dest = transfer_to(
            Tensor(dest_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        src = transfer_to(
            Tensor(src_np),
            PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        # Each shard is (4, 4); update row 0 of each shard.
        buffer_store_slice(dest, src, (slice(0, 1), slice(None)))
        result = dest.to_numpy()
        # Row 0 and row 4 of the global tensor should be 3.0.
        expected = dest_np.copy()
        expected[0, :] = 3.0
        expected[4, :] = 3.0
        np.testing.assert_allclose(result, expected)

    def test_setitem_syntax(self) -> None:
        """Tensor.__setitem__ syntax works for realized tensors."""
        dest_np = np.zeros((4, 4), dtype=np.float32)
        src_np = np.ones((2, 4), dtype=np.float32)
        t = Tensor(dest_np)
        t[1:3, :] = Tensor(src_np)
        expected = dest_np.copy()
        expected[1:3, :] = src_np
        np.testing.assert_allclose(t.to_numpy(), expected)


class ShapeTests(
    _Transpose,
    _Reshape,
    _Permute,
    _Concat,
    _Split,
    _Unsqueeze,
    _Squeeze,
    _Gather,
    _BroadcastTo,
    _Flatten,
    _Stack,
    _Argsort,
    _TopKBottomK,
    _Chunk,
    _RepeatInterleave,
    _Tile,
    _Pad,
    _SliceTensor,
    _Scatter,
    _Outer,
    _Where,
    _GatherNd,
    _ScatterNd,
    _BufferStore,
    _SliceStore,
):
    """Aggregates all shape test classes for thin subclassing."""

    pass
