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
"""Shared test logic for reduction ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_reduction_simulated.py and
functional/test_reduction_multi_gpu.py.

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
from _test_helpers import assert_no_shadowed_tests
from max.experimental.functional import (
    cumsum,
    mean,
    softmax,
    sum,
    transfer_to,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    ShardingError,
    auto_reshard,
)
from max.experimental.tensor import Tensor

# ── Sum ──────────────────────────────────────────────────────────────


class _Sum:
    """Tests for distributed sum reduction."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_sharded_axis(self) -> None:
        """MESH_1D: Sharded(0) + sum along axis=1 preserves placement."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        # Graph ops keep the reduced dim as size 1.
        expected = t_np.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_numerics(self) -> None:
        """MESH_1D with 4 shards: sum along non-sharded axis is correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = t_np.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_partial_passthrough(self) -> None:
        """MESH_1D: Partial placement passes through sum (linear op)."""
        a_np = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Partial(),)
        # to_numpy() materializes (allreduces) the Partial across N devices,
        # so each partial sum of 4.0 becomes 4.0 * N.
        n = self.MESH_1D.num_devices
        result_np = result.to_numpy()
        assert result_np.shape == (4, 1)
        expected = a_np.sum(axis=1, keepdims=True) * n
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)

    def test_axis_none_replicated_ok(self) -> None:
        """MESH_2: sum(Replicated, axis=None) works — all shards identical."""
        t_np = np.arange(8, dtype=np.float32).reshape(2, 4)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_2, (Replicated(),))
        )
        result = sum(t, axis=None)
        # axis=None flattens then reduces → scalar-like (1,) shape
        expected = t_np.sum()
        np.testing.assert_allclose(
            result.to_numpy().item(), expected, rtol=1e-5
        )


# ── Mean ─────────────────────────────────────────────────────────────


class _Mean:
    """Tests for distributed mean reduction."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_mean_non_sharded_axis(self) -> None:
        """MESH_2: Sharded(0) + mean along axis=-1 preserves placement."""
        t_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        t = transfer_to(Tensor(t_np), DeviceMapping(self.MESH_2, (Sharded(0),)))
        result = mean(t, axis=-1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = t_np.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)

    def test_uneven_sharded_axis(self) -> None:
        """A mean over unevenly sharded rows divides by the global count."""
        t_np = np.arange(10, dtype=np.float32).reshape(5, 2)
        t = transfer_to(Tensor(t_np), DeviceMapping(self.MESH_2, (Sharded(0),)))
        with auto_reshard(mode="silent"):
            result = mean(t, axis=0)
        expected = t_np.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


# ── Softmax ──────────────────────────────────────────────────────────


class _Softmax:
    """Tests for distributed softmax."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_softmax_non_sharded_axis(self) -> None:
        """MESH_1D: softmax along non-sharded axis preserves placement."""
        a_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(a_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = softmax(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        arr = result.to_numpy()
        # Compute expected softmax along axis=1
        e = np.exp(a_np - a_np.max(axis=1, keepdims=True))
        expected = e / e.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(arr, expected, rtol=1e-5)


# ── Cumsum ──────────────────────────────────────────────────────────


class _Cumsum:
    """Tests for distributed cumsum."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]
    partial_distinct_fn: ClassVar[Callable[..., Tensor]]

    def test_cumsum_non_sharded_axis(self) -> None:
        """MESH_2: cumsum along non-sharded axis preserves placement."""
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(Tensor(t_np), DeviceMapping(self.MESH_2, (Sharded(0),)))
        result = cumsum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = np.cumsum(t_np, axis=1)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_cumsum_sharded_axis(self) -> None:
        """A prefix sum over the sharded axis continues across shards."""
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(Tensor(t_np), DeviceMapping(self.MESH_2, (Sharded(0),)))
        with auto_reshard(mode="silent"):
            result = cumsum(t, axis=0)
        np.testing.assert_allclose(
            result.to_numpy(), np.cumsum(t_np, axis=0), rtol=1e-5
        )

    def test_cumsum_partial_input_is_not_passed_through(self) -> None:
        """MESH_1D: cumsum has no Partial row, so the default policy raises.

        cumsum is linear, but scanning each fragment and then summing differs
        from summing and then scanning in floating point, so cumsum resolves a
        Partial before the scan instead of passing it through.
        """
        a_np = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        with pytest.raises(ShardingError):
            cumsum(t, axis=1)

    def test_cumsum_partial_input_resolves_under_auto_reshard(self) -> None:
        """MESH_1D: with auto_reshard the Partial is resolved, then scanned."""
        a_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        with auto_reshard(mode="silent"):
            result = cumsum(t, axis=1)
        assert result.placements == (Replicated(),)
        # partial_fn gives every device a copy of a_np, so the Partial sums to
        # num_devices * a_np before the scan.
        n = self.MESH_1D.num_devices
        np.testing.assert_allclose(
            result.to_numpy(), np.cumsum(a_np * n, axis=1), rtol=1e-5
        )

    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
        ],
    )
    def test_cumsum_partial_int_passthrough(self, dtype: np.dtype) -> None:
        """MESH_1D: integral Partial input passes through, bit-exact.

        Each device holds a distinct fragment (rank r holds r+1 copies of
        the base values, with a magnitude that overflows the dtype inside
        both the local scan and the fragment sum), so the test can tell a
        passthrough from a resolve-first mishap. Expected values come
        from an independent modular reference in Python ints.
        """
        # Fragment values chosen so both the local scan and the fragment
        # sum overflow the narrow widths: rank r holds (r+1) * fragment.
        fragment = [
            [3, -(2**40), 11, 5],
            [2, 6, 2**40, 4],
            [1, 9, 8, -(2**40)],
            [2**40, -2, 7, 12],
        ]

        is_unsigned = dtype.name.startswith("uint")

        def wrap(value: int) -> int:
            # Reduce into the dtype's range: two's-complement for signed,
            # plain modular for unsigned. NumPy raises on a Python int
            # that overflows the dtype, and the wrapped fragment is what
            # a device would hold.
            modulus = 1 << (8 * dtype.itemsize)
            if is_unsigned:
                return value % modulus
            half = modulus // 2
            return (value + half) % modulus - half

        def fn(rank: int) -> np.ndarray:
            return np.array(
                [[wrap((rank + 1) * v) for v in row] for row in fragment],
                dtype=dtype,
            )

        t = self.partial_distinct_fn(fn, self.MESH_1D)
        result = cumsum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Partial(),)

        # Reference: exact Python ints (no NumPy promotion), reduced to
        # the dtype's two's-complement range after the scan.
        n = self.MESH_1D.num_devices
        modulus = 1 << (8 * dtype.itemsize)
        resolved = []
        for row in fragment:
            resolved_row = []
            for v in row:
                total = 0
                for r in range(n):
                    total += (r + 1) * v
                resolved_row.append(total)
            resolved.append(resolved_row)
        wrapped = []
        for row in resolved:
            prefix = 0
            out_row = []
            for value in row:
                prefix = (prefix + value) % modulus
                out_row.append(wrap(prefix))
            wrapped.append(out_row)
        expected = np.array(wrapped, dtype=dtype)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize(
        "exclusive,reverse",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_cumsum_partial_int_exclusive_reverse(
        self, exclusive: bool, reverse: bool
    ) -> None:
        """MESH_1D: exclusive/reverse variants pass through and stay exact.

        Odd length exercises both flags' boundary positions; a distinct
        per-rank fragment keeps rank identity observable.
        """
        length = 7

        def fn(rank: int) -> np.ndarray:
            return (np.arange(1, length + 1) * (rank + 2)).astype(np.int32)

        t = self.partial_distinct_fn(fn, self.MESH_1D)
        result = cumsum(t, axis=0, exclusive=exclusive, reverse=reverse)
        assert result.placements == (Partial(),)

        # The contract: resolved results agree bit-for-bit with resolve-first
        # execution. Scan the summed fragments as one replicated tensor
        # through the same op and compare.
        n = self.MESH_1D.num_devices
        resolved_np = np.sum(
            np.stack([fn(rank) for rank in range(n)]), axis=0
        ).astype(np.int32)
        resolved = transfer_to(
            Tensor(resolved_np), DeviceMapping(self.MESH_1D, (Replicated(),))
        )
        expected = cumsum(
            resolved, axis=0, exclusive=exclusive, reverse=reverse
        )
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())


# ── Combined base class ─────────────────────────────────────────────


class ReductionTests(_Sum, _Mean, _Softmax, _Cumsum):
    """Aggregates all reduction test classes for thin subclassing."""


assert_no_shadowed_tests(ReductionTests)
