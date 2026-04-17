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
"""Shared test logic for elementwise distributed ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_elementwise_simulated.py and
functional/test_elementwise_multi_gpu.py.

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
from max.dtype import DType
from max.experimental.distributed_functional import (
    add,
    cast,
    div,
    exp,
    mul,
    negate,
    relu,
    silu,
    transfer_to,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

# ── TestUnaryNonlinear (relu) ────────────────────────────────────────────


class _UnaryNonlinear:
    """Tests for unary non-linear elementwise ops (relu)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_relu_sharded(self) -> None:
        arr = np.array(
            [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
            dtype=np.float32,
        )
        t = transfer_to(
            Tensor(arr),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = relu(t)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        expected = np.maximum(arr, 0.0)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_relu_partial_auto_reduces(self) -> None:
        arr = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_1D, (Partial(),))
        result = relu(t)
        # Non-linear op auto-reduces Partial -> Replicated, then applies relu.
        assert result.placements == (Replicated(),)
        expected = np.maximum(arr * 4, 0.0)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_relu_partial_auto_reduces_ones(self) -> None:
        arr = np.ones((2, 2), dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_1D, (Partial(),))
        # auto_reduce_partial is now a no-op; transfer_to is deterministic.
        result = relu(t)
        assert result.placements == (Replicated(),)


# ── TestUnaryLinear (negate) ─────────────────────────────────────────────


class _UnaryLinear:
    """Tests for unary linear elementwise ops (negate)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_negate_sharded(self) -> None:
        arr = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        t = transfer_to(
            Tensor(arr),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = negate(t)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), -arr, rtol=1e-5)

    def test_negate_partial_passthrough(self) -> None:
        arr = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_1D, (Partial(),))
        result = negate(t)
        # Linear op: Partial passes through without reduction.
        assert result.placements == (Partial(),)
        # Each shard holds -arr; full reduce would give -arr * 4.
        expected = -arr * 4
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_negate_partial_passthrough_explicit(self) -> None:
        arr = np.full((4, 8), 3.0, dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_1D, (Partial(),))
        result = negate(t)
        assert result.placements == (Partial(),)
        # negate is linear: each shard is -3.0, sum of 4 shards = -12.0
        expected = -arr * 4
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


# ── TestBinaryNonlinear (mul) ────────────────────────────────────────────


class _BinaryNonlinear:
    """Tests for binary non-linear elementwise ops (mul)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_mul_sharded(self) -> None:
        a = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        b = np.array(
            [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]], dtype=np.float32
        )
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(b),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = mul(ta, tb)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a * b, rtol=1e-5)

    def test_mul_incompatible_raises(self) -> None:
        a = np.ones((4, 8), dtype=np.float32)
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(1),
                    Replicated(),
                ),
            ),
        )
        with pytest.raises(ValueError, match="incompatible"):
            mul(ta, tb)

    def test_mul_partial_auto_reduces(self) -> None:
        a = np.full((4, 8), 2.0, dtype=np.float32)
        b = np.full((4, 8), 3.0, dtype=np.float32)
        ta = self.partial_fn(a, self.MESH_1D, (Partial(),))
        tb = self.partial_fn(b, self.MESH_1D, (Partial(),))
        result = mul(ta, tb)
        # Non-linear: both partials auto-reduce first, then mul.
        assert result.placements == (Replicated(),)
        expected = (a * 4) * (b * 4)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_mul_partial_auto_reduces_same_input(self) -> None:
        a = np.full((4, 8), 2.0, dtype=np.float32)
        ta = self.partial_fn(a, self.MESH_1D, (Partial(),))
        tb = self.partial_fn(a, self.MESH_1D, (Partial(),))
        # auto_reduce_partial is now a no-op; transfer_to is deterministic.
        result = mul(ta, tb)
        assert result.placements == (Replicated(),)


# ── TestBinaryLinear (add) ───────────────────────────────────────────────


class _BinaryLinear:
    """Tests for binary linear elementwise ops (add)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_add_sharded(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.ones((2, 4), dtype=np.float32) * 10
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(b),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = add(ta, tb)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a + b, rtol=1e-5)

    def test_add_incompatible_raises(self) -> None:
        a = np.ones((4, 8), dtype=np.float32)
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(1),
                    Replicated(),
                ),
            ),
        )
        with pytest.raises(ValueError, match="incompatible"):
            add(ta, tb)

    def test_add_pp_passthrough(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        b = np.full((2, 4), 2.0, dtype=np.float32)
        ta = self.partial_fn(a, self.MESH_1D, (Partial(),))
        tb = self.partial_fn(b, self.MESH_1D, (Partial(),))
        result = add(ta, tb)
        # Linear: both Partial -> passthrough (Partial + Partial = Partial).
        assert result.placements == (Partial(),)
        # Each shard = a + b = 3.0; 4 shards -> full value = 12.0
        expected = (a + b) * 4
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_add_pr_auto_reduces(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        b = np.full((2, 4), 2.0, dtype=np.float32)
        ta = self.partial_fn(a, self.MESH_1D, (Partial(),))
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = add(ta, tb)
        # Linear mixed (Partial + Replicated): auto-reduces the Partial.
        assert result.placements == (Replicated(),)
        expected = (a * 4) + b
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_add_pr_auto_reduces_deterministic(self) -> None:
        a = np.ones((2, 2), dtype=np.float32)
        b = np.full((2, 2), 2.0, dtype=np.float32)
        ta = self.partial_fn(a, self.MESH_1D, (Partial(),))
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        # auto_reduce_partial is now a no-op; transfer_to is deterministic.
        result = add(ta, tb)
        assert result.placements == (Replicated(),)


# ── TestBroadcast ────────────────────────────────────────────────────────


class _Broadcast:
    """Tests for broadcast semantics in binary elementwise ops."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_broadcast_rhs_lower_rank(self) -> None:
        # (2,4) + (4,) with RHS replicated on a 2-device mesh
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        ta = transfer_to(
            Tensor(a), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = add(ta, tb)
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a + b, rtol=1e-5)

    def test_broadcast_lhs_lower_rank(self) -> None:
        # (4,) + (2,4)
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.arange(8, dtype=np.float32).reshape(2, 4)
        ta = transfer_to(
            Tensor(a), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = add(ta, tb)
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a + b, rtol=1e-5)

    def test_broadcast_mul_rank_mismatch(self) -> None:
        # (3,4) * (4,) — mul with broadcast
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        ta = transfer_to(
            Tensor(a), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = mul(ta, tb)
        assert tuple(result.shape) == (3, 4)
        np.testing.assert_allclose(result.to_numpy(), a * b, rtol=1e-5)

    def test_broadcast_same_rank(self) -> None:
        # (2,4) + (1,4) — broadcast on dim 0, sharded on dim 1 with 2D mesh
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(b),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = add(ta, tb)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a + b, rtol=1e-5)


# ── TestCast ─────────────────────────────────────────────────────────────


class _Cast:
    """Tests for distributed cast op."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_cast_preserves_placement(self) -> None:
        arr = np.arange(8, dtype=np.float32).reshape(2, 4)
        t = transfer_to(
            Tensor(arr),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = cast(t, DType.bfloat16)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        # Cast back to float32 and check values
        result_f32 = cast(result, DType.float32)
        np.testing.assert_allclose(result_f32.to_numpy(), arr, rtol=1e-2)

    def test_cast_partial_auto_reduces(self) -> None:
        arr = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_1D, (Partial(),))
        result = cast(t, DType.bfloat16)
        # Cast is non-linear: auto-reduces Partial first.
        assert result.placements == (Replicated(),)
        result_f32 = cast(result, DType.float32)
        np.testing.assert_allclose(result_f32.to_numpy(), arr * 4, rtol=1e-2)


# ── TestSmoke ────────────────────────────────────────────────────────────


class _Smoke:
    """Smoke tests for various elementwise ops."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_exp(self) -> None:
        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32
        )
        t = transfer_to(
            Tensor(arr),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = exp(t)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), np.exp(arr), rtol=1e-5)

    def test_silu(self) -> None:
        arr = np.array(
            [[-1.0, 0.0, 1.0, 2.0], [-2.0, -0.5, 0.5, 3.0]], dtype=np.float32
        )
        t = transfer_to(
            Tensor(arr),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = silu(t)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        expected = arr * (1.0 / (1.0 + np.exp(-arr)))
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_div(self) -> None:
        a = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
        b = np.full((2, 4), 2.0, dtype=np.float32)
        ta = transfer_to(
            Tensor(a),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        tb = transfer_to(
            Tensor(b),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = div(ta, tb)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), a / b, rtol=1e-5)

    def test_chain_add_negate(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.ones((2, 4), dtype=np.float32) * 10
        ta = transfer_to(
            Tensor(a), PlacementMapping(self.MESH_1D, (Sharded(1),))
        )
        tb = transfer_to(
            Tensor(b), PlacementMapping(self.MESH_1D, (Sharded(1),))
        )
        result = negate(add(ta, tb))
        assert result.placements == (Sharded(1),)
        assert tuple(result.shape) == (2, 4)
        np.testing.assert_allclose(result.to_numpy(), -(a + b), rtol=1e-5)


# ── Replicated + Sharded mixed-placement ───────────────────────────────


class _MixedPlacement:
    """Tests for binary ops with Replicated + Sharded operands.

    Validates that align_shards correctly splits Replicated full-sized
    values to match Sharded local sizes before per-shard dispatch.
    """

    MESH_2: ClassVar[DeviceMesh]

    def test_add_replicated_plus_sharded(self) -> None:
        """add(Replicated, Sharded(0)) produces correct Sharded(0) result."""
        a_np = np.ones((4, 2), dtype=np.float32) * 10.0
        b_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = add(a, b)
        assert result.placements == (Sharded(0),)
        assert tuple(result.shape) == (4, 2)
        np.testing.assert_allclose(result.to_numpy(), a_np + b_np, rtol=1e-5)

    def test_mul_sharded_times_replicated(self) -> None:
        """mul(Sharded(0), Replicated) produces correct Sharded(0) result."""
        a_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        b_np = np.ones((4, 2), dtype=np.float32) * 3.0
        a = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = mul(a, b)
        assert result.placements == (Sharded(0),)
        assert tuple(result.shape) == (4, 2)
        np.testing.assert_allclose(result.to_numpy(), a_np * b_np, rtol=1e-5)


class ElementwiseTests(
    _UnaryNonlinear,
    _UnaryLinear,
    _BinaryNonlinear,
    _BinaryLinear,
    _Broadcast,
    _Cast,
    _Smoke,
    _MixedPlacement,
):
    """Aggregates all elementwise test classes for thin subclassing."""

    pass
