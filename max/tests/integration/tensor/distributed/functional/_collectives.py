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
"""Shared test logic for collective ops and transfer_to.

DO NOT run this file directly — it contains base classes that are
subclassed by test_collectives_simulated.py and test_collectives_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    partial_fn: Callable   — make_partial (CPU) or make_partial (GPU)

Same-mesh transitions, per mesh axis:

    Source → Target          | Collective
    -------------------------|------------------------------------------
    Same → Same              | none
    Partial → Replicated     | allreduce
    Partial → Sharded(dim)   | reduce_scatter
    Sharded → Replicated     | allgather
    Replicated → Sharded     | local split, no communication
    Replicated → Partial     | one copy, zeros elsewhere
    Sharded(i) → Sharded(j)  | all_to_all for equal static shards, else
                             | allgather(i) and local split(j)

Cross-mesh: between meshes of one shape each shard moves to the device at
its position; otherwise the tensor gathers, moves, and scatters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.functional import (
    all_to_all,
    allgather,
    allgather_rms_norm,
    allreduce_sum,
    collective_ops,
    reduce_scatter,
    reduce_scatter_rms_norm,
    transfer_to,
)
from max.nn import Module
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    TensorLayout,
)
from max.tensor.sharding.placements import local_shard_shape_from_global

_COLLECTIVES = ("allreduce_sum", "allgather", "reduce_scatter", "all_to_all")


@contextmanager
def _collectives_called() -> Iterator[list[str]]:
    """Records the collectives transfer_to calls in the block, by name."""
    with ExitStack() as stack:
        mocks = [
            stack.enter_context(
                patch.object(
                    collective_ops, name, wraps=getattr(collective_ops, name)
                )
            )
            for name in _COLLECTIVES
        ]
        names: list[str] = []
        yield names
        names.extend(
            name
            for name, mock in zip(_COLLECTIVES, mocks, strict=True)
            for _ in range(mock.call_count)
        )


class _SymbolicTransfers(
    Module[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]
):
    """Moves a row-split and a Partial input to Replicated and to a split."""

    def forward(
        self, rows: Tensor, partial: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mesh = rows.mesh
        replicated = DeviceMapping(mesh, (Replicated(),))
        return (
            rows.to(replicated),
            rows.to(DeviceMapping(mesh, (Sharded(1),))),
            partial.to(replicated),
            partial.to(DeviceMapping(mesh, (Sharded(0),))),
        )


class CollectivesTests:
    """Base class with collective test methods.

    Subclass must set MESH_1D, MESH_2D, and partial_fn.
    """

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    # ─── Helper: validate ALL shards numerically ─────────────────────

    @staticmethod
    def _compute_shard(
        full: np.ndarray,
        device_idx: int,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> np.ndarray:
        """Compute what shard device_idx should hold given placements."""
        result = full
        coords: list[int] = []
        remaining = device_idx
        for ax in range(len(mesh.mesh_shape) - 1, -1, -1):
            coords.insert(0, remaining % mesh.mesh_shape[ax])
            remaining //= mesh.mesh_shape[ax]

        for ax, p in enumerate(placements):
            if isinstance(p, Sharded):
                n = mesh.mesh_shape[ax]
                coord = coords[ax]
                dim_size = result.shape[p.axis]
                chunk = dim_size // n
                start = coord * chunk
                end = start + chunk
                slices = [slice(None)] * result.ndim
                slices[p.axis] = slice(start, end)
                result = result[tuple(slices)]
        return result

    def _assert_shards(
        self,
        result: Tensor,
        expected_full: np.ndarray,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> None:
        """Validate placements AND all shards numerically."""
        assert result.placements == placements
        for i in range(mesh.num_devices):
            actual = result.local_shards[i].to_numpy()
            expected = self._compute_shard(expected_full, i, mesh, placements)
            np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # ─── Helper: create mixed Partial+Sharded tensor ────────────────

    def _make_partial_sharded(
        self,
        data: np.ndarray,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> Tensor:
        """Create a tensor with mixed Partial+Sharded placements.

        Shards along any Sharded axes first, then relabels Partial axes.
        (make_partial only handles pure-Partial placements.)
        """
        # Build a shard placement replacing Partial with Replicated.
        shard_placements = tuple(
            Replicated() if isinstance(p, Partial) else p for p in placements
        )
        sharded = transfer_to(
            Tensor(data), DeviceMapping(mesh, tuple(shard_placements))
        )
        return Tensor._from_shards(
            tuple(s.driver_tensor for s in sharded.local_shards),
            DeviceMapping(mesh, placements),
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 1: Atomic collective ops (allreduce, allgather, reduce_scatter)
    # ═════════════════════════════════════════════════════════════════════

    # ── allreduce_sum ────────────────────────────────────────────────

    def test_allreduce_1d(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    def test_allreduce_2d_tp_axis(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        t = self.partial_fn(a, self.MESH_2D, (Replicated(), Partial()))
        result = allreduce_sum(t, mesh_axis=1)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 2, rtol=1e-5
            )

    def test_allreduce_2d_dp_axis(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Replicated()))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 2, rtol=1e-5
            )

    def test_allreduce_shard_values(self) -> None:
        """All 4 shards hold the reduced value after allreduce."""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    # ── allgather ────────────────────────────────────────────────────

    def test_allgather_1d_axis0(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_1d_axis1(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(2, 16)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(1),))
        )
        result = allgather(sharded, tensor_axis=1, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_2d_tp_axis(self) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        sharded = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(0),
                ),
            ),
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=1)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_2d_dp_axis(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        sharded = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    # ── reduce_scatter ───────────────────────────────────────────────

    def test_reduce_scatter_1d(self) -> None:
        a = np.ones((8, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        assert result.placements == (Sharded(0),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(),
                np.full((2, 4), 4.0, dtype=np.float32),
                rtol=1e-5,
            )

    def test_reduce_scatter_roundtrip(self) -> None:
        a = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        scattered = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        gathered = allgather(scattered, tensor_axis=0, mesh_axis=0)
        np.testing.assert_allclose(
            gathered.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_reduce_scatter_shard_values(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        assert result.placements == (Sharded(0),)
        expected_full = a * 4
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(),
                expected_full[i : i + 1],
                rtol=1e-5,
            )

    # ── fused collective + RMS norm ──────────────────────────────────

    _NORM_EPS = 1e-6

    def _norm_dtype(self) -> DType:
        # The fused kernels are bfloat16-only; a mesh without accelerators
        # takes the simulated path, which runs in float32 on CPU.
        if all(isinstance(d, Accelerator) for d in self.MESH_1D.devices):
            return DType.bfloat16
        return DType.float32

    def _assert_close_in(
        self, actual: Tensor, expected: np.ndarray, dtype: DType
    ) -> None:
        tol = 3e-2 if dtype == DType.bfloat16 else 1e-4
        np.testing.assert_allclose(
            actual.cast(DType.float32).to_numpy(),
            expected,
            rtol=tol,
            atol=tol,
        )

    def _replicated_gamma(self, gamma: np.ndarray, dtype: DType) -> Tensor:
        return transfer_to(
            Tensor(gamma), DeviceMapping(self.MESH_1D, (Replicated(),))
        ).cast(dtype)

    @classmethod
    def _rms_norm(cls, x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + cls._NORM_EPS)
        return x / rms * gamma

    @pytest.mark.compile_on_gpu("the interpreter has no fused collective")
    def test_reduce_scatter_rms_norm_1d(self) -> None:
        mesh = self.MESH_1D
        dtype = self._norm_dtype()
        rng = np.random.default_rng(0)
        fragments = [
            rng.standard_normal((8, 64)).astype(np.float32)
            for _ in range(mesh.num_devices)
        ]
        gamma = rng.standard_normal(64).astype(np.float32)

        shards = []
        for fragment, device in zip(fragments, mesh.devices, strict=True):
            shard = Tensor(fragment)
            if shard.device != device:
                shard = Tensor(storage=shard.driver_tensor.to(device))
            shards.append(shard.cast(dtype).driver_tensor)
        t = Tensor._from_shards(
            tuple(shards), DeviceMapping(mesh, (Partial(),))
        )

        normed, summed = reduce_scatter_rms_norm(
            t, self._replicated_gamma(gamma, dtype), self._NORM_EPS
        )

        assert normed.placements == (Sharded(0),)
        assert summed.placements == (Sharded(0),)
        total = np.sum(fragments, axis=0)
        for i in range(mesh.num_devices):
            rows = total[2 * i : 2 * i + 2]
            self._assert_close_in(summed.local_shards[i], rows, dtype)
            self._assert_close_in(
                normed.local_shards[i], self._rms_norm(rows, gamma), dtype
            )

    @pytest.mark.compile_on_gpu("the interpreter has no fused collective")
    def test_allgather_rms_norm_1d(self) -> None:
        mesh = self.MESH_1D
        dtype = self._norm_dtype()
        rng = np.random.default_rng(1)
        full = rng.standard_normal((8, 64)).astype(np.float32)
        gamma = rng.standard_normal(64).astype(np.float32)
        sharded = transfer_to(
            Tensor(full), DeviceMapping(mesh, (Sharded(0),))
        ).cast(dtype)

        normed, gathered = allgather_rms_norm(
            sharded, self._replicated_gamma(gamma, dtype), self._NORM_EPS
        )

        assert normed.placements == (Replicated(),)
        assert gathered.placements == (Replicated(),)
        for i in range(mesh.num_devices):
            self._assert_close_in(gathered.local_shards[i], full, dtype)
            self._assert_close_in(
                normed.local_shards[i], self._rms_norm(full, gamma), dtype
            )

    def test_reduce_scatter_rms_norm_rejects_sharded(self) -> None:
        sharded = transfer_to(
            Tensor(np.ones((8, 4), dtype=np.float32)),
            DeviceMapping(self.MESH_1D, (Sharded(0),)),
        )
        gamma = self._replicated_gamma(
            np.ones(4, dtype=np.float32), DType.float32
        )
        with pytest.raises(ValueError, match="Partial"):
            reduce_scatter_rms_norm(sharded, gamma, self._NORM_EPS)

    # ═════════════════════════════════════════════════════════════════════
    #  Section 2: allreduce_sum / allgather multi-axis tests
    # ═════════════════════════════════════════════════════════════════════

    def test_allreduce_sum_1d_partial(self) -> None:
        """Partial on a 1D mesh — single allreduce resolves to Replicated."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_allgather_1d_sharded(self) -> None:
        """Sharded on a 1D mesh — single allgather resolves to Replicated."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.local_shards[0].to_numpy(), t_np, rtol=1e-5
        )

    def test_allreduce_sum_2d_both_axes(self) -> None:
        """Partial on both axes — two allreduces, one per axis."""
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        t = allreduce_sum(t, mesh_axis=0)
        t = allreduce_sum(t, mesh_axis=1)
        assert t.placements == (Replicated(), Replicated())
        np.testing.assert_allclose(
            t.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_allreduce_sum_preserves_sharded_axis(self) -> None:
        """(Sharded, Partial) — allreduce on axis 1 leaves axis 0 Sharded."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Sharded(0), Partial()))
        result = allreduce_sum(t, mesh_axis=1)
        assert result.placements[0] == Sharded(0)
        assert result.placements[1] == Replicated()

    # ═════════════════════════════════════════════════════════════════════
    #  Section 3: scatter (distribute) — non-distributed → distributed
    # ═════════════════════════════════════════════════════════════════════

    def test_scatter_sharded_1d(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        result = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_scatter_replicated_1d(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        result = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Replicated(),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_scatter_2d_mesh(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = transfer_to(
            Tensor(t_np),
            DeviceMapping(self.MESH_2D, (Replicated(), Sharded(1))),
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    def test_scatter_2d_both_sharded(self) -> None:
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        result = transfer_to(
            Tensor(t_np),
            DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(1))),
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    def test_scatter_roundtrip(self) -> None:
        original = np.arange(32, dtype=np.float32).reshape(8, 4)
        distributed = transfer_to(
            Tensor(original), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        gathered = allgather(distributed, tensor_axis=0, mesh_axis=0)
        np.testing.assert_allclose(
            gathered.local_shards[0].to_numpy(), original, rtol=1e-5
        )

    def test_scatter_rejects_partial(self) -> None:
        with pytest.raises(ValueError, match="Partial"):
            transfer_to(
                Tensor(np.ones((2, 4), dtype=np.float32)),
                DeviceMapping(self.MESH_1D, (Partial(),)),
            )

    def test_scatter_rejects_splitting_a_size_one_axis(self) -> None:
        with pytest.raises(ShardingError, match="static extent 1"):
            transfer_to(
                Tensor(np.ones((1, 8), dtype=np.float32)),
                DeviceMapping(self.MESH_1D, (Sharded(0),)),
            )

    def test_all_to_all_reads_a_negative_sharded_axis(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(-1),))
        )
        swapped = all_to_all(t, split_axis=0, concat_axis=-1)
        self._assert_shards(swapped, t_np, self.MESH_1D, (Sharded(0),))

    def test_all_to_all_rejects_an_indivisible_split(self) -> None:
        t = transfer_to(
            Tensor(np.ones((8, 3), dtype=np.float32)),
            DeviceMapping(self.MESH_1D, (Sharded(0),)),
        )
        with pytest.raises(ValueError, match="multiple of the group size"):
            all_to_all(t, split_axis=1, concat_axis=0)

    @pytest.mark.parametrize(
        "source,target", [(-1, 1), (1, -1)], ids=["S-1->S1", "S1->S-1"]
    )
    def test_transfer_between_negative_and_positive_axes(
        self, source: int, target: int
    ) -> None:
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(source),))
        )
        moved = transfer_to(t, DeviceMapping(self.MESH_1D, (Sharded(target),)))
        assert moved.placements == (Sharded(target),)
        expected = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(1),))
        )
        for actual, want in zip(
            moved.local_shards, expected.local_shards, strict=True
        ):
            np.testing.assert_array_equal(actual.to_numpy(), want.to_numpy())

    def test_scatter_splits_a_negative_and_positive_axis_once(self) -> None:
        t_np = np.arange(40, dtype=np.float32).reshape(4, 10)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(self.MESH_2D, (Sharded(1), Sharded(-1))),
        )
        expected = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_2D, (Sharded(1), Sharded(1)))
        )
        for actual, want in zip(
            t.local_shards, expected.local_shards, strict=True
        ):
            np.testing.assert_array_equal(actual.to_numpy(), want.to_numpy())

    def test_swap_on_one_mesh_axis_checks_its_own_group(self) -> None:
        # On distinct devices the groups launch as one kernel over the mesh.
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(self.MESH_2D, (Replicated(), Sharded(0))),
        )
        moved = transfer_to(
            t, DeviceMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        )
        self._assert_shards(
            moved, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 4: materialize / to_numpy
    # ═════════════════════════════════════════════════════════════════════

    def test_materialize_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sharded.materialize()
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), t_np, rtol=1e-5)

    def test_materialize_partial(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = t.materialize()
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), a * 4, rtol=1e-5)

    def test_materialize_noop(self) -> None:
        t = Tensor(np.ones((2, 4), dtype=np.float32))
        result = t.materialize()
        assert not result.is_distributed

    def test_to_numpy_distributed(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        np.testing.assert_allclose(sharded.to_numpy(), t_np, rtol=1e-5)

    def test_to_numpy_non_distributed(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        np.testing.assert_allclose(Tensor(t_np).to_numpy(), t_np, rtol=1e-5)

    # ═════════════════════════════════════════════════════════════════════
    #  Section 5: transfer_to — same mesh, 1D
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_3d_tensor_sharded_axis0_to_axis2(self) -> None:
        t_np = np.arange(96, dtype=np.float32).reshape(4, 6, 4)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        result = transfer_to(t, DeviceMapping(self.MESH_1D, (Sharded(2),)))
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(2),))

    def test_redistribute_3d_tensor_sharded_to_replicated(self) -> None:
        t_np = np.arange(96, dtype=np.float32).reshape(2, 8, 6)
        t = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(1),))
        )
        result = transfer_to(t, DeviceMapping(self.MESH_1D, (Replicated(),)))
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    # ═════════════════════════════════════════════════════════════════════
    #  Section 6: transfer_to — same mesh, 2D, single-axis change
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_2d_partial_partial_to_replicated_replicated(
        self,
    ) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        result = transfer_to(
            t, DeviceMapping(self.MESH_2D, (Replicated(), Replicated()))
        )
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    def test_redistribute_2d_replicated_sharded_to_sharded_replicated(
        self,
    ) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = transfer_to(
            t, DeviceMapping(self.MESH_2D, (Sharded(0), Replicated()))
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_redistribute_2d_partial_sharded_to_replicated_sharded(
        self,
    ) -> None:
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        result = transfer_to(
            t, DeviceMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        )
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_sharded_to_sharded_different_axis(
        self,
    ) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(0),
                ),
            ),
        )
        result = transfer_to(
            t, DeviceMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 7: transfer_to — same mesh, 2D, BOTH axes change
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_2d_S0_S1_to_R_R(self) -> None:
        """(Sharded(0), Sharded(1)) → (Replicated, Replicated)."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_S0_S1_to_S1_S0(self) -> None:
        """(Sharded(0), Sharded(1)) → (Sharded(1), Sharded(0))."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Sharded(1), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(1), Sharded(0))
        )

    def test_redistribute_2d_S0_S1_to_R_S0(self) -> None:
        """(Sharded(0), Sharded(1)) → (Replicated, Sharded(0))."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_S0_R_to_R_S0(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_S0_R_to_R_S1(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    def test_redistribute_2d_S0_R_to_S0_S1(self) -> None:
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    # ── Mixed Partial+Sharded source, both axes change ───────────────

    def test_redistribute_2d_P_S0_to_R_R(self) -> None:
        """(Partial, Sharded(0)) → (Replicated, Replicated)."""
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        target = DeviceMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_P_S0_to_S0_R(self) -> None:
        """(Partial, Sharded(0)) → (Sharded(0), Replicated).

        Partial on dp can't reduce-scatter to Sharded(0) because tp already
        shards axis 0. Falls back to allreduce + allgather + local_split.
        """
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        target = DeviceMapping(self.MESH_2D, (Sharded(0), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_redistribute_2d_S0_P_to_R_R(self) -> None:
        """(Sharded(0), Partial) → (Replicated, Replicated)."""
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Sharded(0), Partial()))
        target = DeviceMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on tp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_S0_P_to_S0_S1(self) -> None:
        """(Sharded(0), Partial) → (Sharded(0), Sharded(1))."""
        a = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = self._make_partial_sharded(a, self.MESH_2D, (Sharded(0), Partial()))
        target = DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on tp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 8: transfer_to — cross-mesh transfers
    # ═════════════════════════════════════════════════════════════════════

    def test_cross_mesh_sharded_to_single_device(self) -> None:
        """4-device sharded → 1-device (materialize)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        single_mesh = DeviceMesh.single(self.MESH_1D.devices[0])
        target = DeviceMapping(single_mesh, (Replicated(),))
        result = transfer_to(sharded, target)
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), t_np, rtol=1e-5)

    def test_cross_mesh_single_to_sharded(self) -> None:
        """1-device → 4-device sharded (weight loading)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = Tensor(t_np)
        target = DeviceMapping(self.MESH_1D, (Sharded(0),))
        result = transfer_to(t, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_cross_mesh_single_to_replicated(self) -> None:
        """1-device → 4-device replicated."""
        t_np = np.ones((4, 4), dtype=np.float32)
        t = Tensor(t_np)
        target = DeviceMapping(self.MESH_1D, (Replicated(),))
        result = transfer_to(t, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_cross_mesh_partial_to_single_device(self) -> None:
        """4-device partial → 1-device (allreduce + materialize)."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        single_mesh = DeviceMesh.single(self.MESH_1D.devices[0])
        target = DeviceMapping(single_mesh, (Replicated(),))
        result = transfer_to(t, target)
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), a * 4, rtol=1e-5)

    def test_cross_mesh_1d_to_2d(self) -> None:
        """1D sharded(4) → 2D (2,2) with (Sharded(0), Replicated)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded_1d = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        target = DeviceMapping(self.MESH_2D, (Sharded(0), Replicated()))
        result = transfer_to(sharded_1d, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_cross_mesh_2d_to_1d(self) -> None:
        """2D (Sharded(0), Replicated) → 1D Sharded(0)."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        sharded_2d = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = DeviceMapping(self.MESH_1D, (Sharded(0),))
        result = transfer_to(sharded_2d, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    # ═════════════════════════════════════════════════════════════════════
    #  Section 9: roundtrip tests — transfer_to A→B→A = identity
    # ═════════════════════════════════════════════════════════════════════

    def test_roundtrip_sharded_replicated_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        replicated = transfer_to(
            sharded, DeviceMapping(self.MESH_1D, (Replicated(),))
        )
        back = transfer_to(
            replicated, DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(back, t_np, self.MESH_1D, (Sharded(0),))

    def test_roundtrip_partial_to_replicated_to_sharded(self) -> None:
        a = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        replicated = transfer_to(
            t, DeviceMapping(self.MESH_1D, (Replicated(),))
        )
        sharded = transfer_to(
            replicated, DeviceMapping(self.MESH_1D, (Sharded(0),))
        )
        expected = a * 4
        self._assert_shards(sharded, expected, self.MESH_1D, (Sharded(0),))

    def test_roundtrip_2d_sharded_swap_axes(self) -> None:
        """(S0, S1) → (S1, S0) → (S0, S1) = identity."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        original = transfer_to(
            Tensor(t_np),
            DeviceMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        swapped = transfer_to(
            original, DeviceMapping(self.MESH_2D, (Sharded(1), Sharded(0)))
        )
        back = transfer_to(
            swapped, DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        )
        self._assert_shards(back, t_np, self.MESH_2D, (Sharded(0), Sharded(1)))

    # ─── Collectives over several mesh axes ─────────────────────────

    @staticmethod
    def _expected_shard(
        full: np.ndarray,
        device_idx: int,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> np.ndarray:
        """Returns the part of ``full`` that ``device_idx`` holds.

        The mesh axes that shard a tensor axis split it once over all their
        devices, in row-major order, with the larger pieces first, as
        ``np.array_split`` does.
        """
        result = full
        for axis in {p.axis for p in placements if isinstance(p, Sharded)}:
            group = [
                ax for ax, p in enumerate(placements) if p == Sharded(axis)
            ]
            chunks = np.array_split(
                result, math.prod(mesh.mesh_shape[ax] for ax in group), axis
            )
            result = chunks[mesh.device_coord(device_idx, group)]
        return result

    def _assert_split_shards(
        self, result: Tensor, full: np.ndarray, placements: tuple[Any, ...]
    ) -> None:
        mesh = self.MESH_2D
        assert result.placements == placements
        for i in range(mesh.num_devices):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(),
                self._expected_shard(full, i, mesh, placements),
                rtol=1e-5,
            )

    @pytest.mark.parametrize("rows", [8, 9, 10])
    def test_split_along_two_mesh_axes_matches_shape_model(
        self, rows: int
    ) -> None:
        a = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        placements = (Sharded(0), Sharded(0))
        placed = transfer_to(Tensor(a), DeviceMapping(self.MESH_2D, placements))
        self._assert_split_shards(placed, a, placements)
        model = local_shard_shape_from_global(
            [rows, 3], self.MESH_2D, placements
        )
        assert [
            [int(d) for d in shard.shape] for shard in placed.local_shards
        ] == [[int(d) for d in shape] for shape in model]

    @pytest.mark.parametrize("rows", [8, 9, 10])
    def test_allreduce_over_two_mesh_axes(self, rows: int) -> None:
        a = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        result = allreduce_sum(t, mesh_axis=("dp", "tp"))
        self._assert_split_shards(result, a * 4, (Replicated(), Replicated()))

    @pytest.mark.parametrize("rows", [8, 9, 10])
    def test_allgather_over_two_mesh_axes(self, rows: int) -> None:
        a = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        placed = transfer_to(
            Tensor(a), DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(0)))
        )
        result = allgather(placed, tensor_axis=0, mesh_axis=("dp", "tp"))
        self._assert_split_shards(result, a, (Replicated(), Replicated()))

    @pytest.mark.parametrize("rows", [8, 9, 10])
    def test_partial_to_split_along_two_mesh_axes(self, rows: int) -> None:
        a = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        placements = (Sharded(0), Sharded(0))
        result = transfer_to(t, DeviceMapping(self.MESH_2D, placements))
        self._assert_split_shards(result, a * 4, placements)

    @pytest.mark.parametrize(
        "target",
        [
            (Replicated(), Sharded(0)),
            (Sharded(0), Replicated()),
            (Replicated(), Replicated()),
            (Sharded(1), Sharded(0)),
        ],
    )
    @pytest.mark.parametrize("rows", [8, 9, 10])
    def test_unsplit_from_two_mesh_axes(
        self, rows: int, target: tuple[Any, ...]
    ) -> None:
        a = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
        placed = transfer_to(
            Tensor(a), DeviceMapping(self.MESH_2D, (Sharded(0), Sharded(0)))
        )
        result = transfer_to(placed, DeviceMapping(self.MESH_2D, target))
        # Gathering some mesh axes of a split does not rebalance the other
        # pieces: 10 rows split as 3, 3, 2, 2 gather into 6 and 4 rows, not
        # 5 and 5, and the layout must report those sizes.
        assert result.placements == target
        np.testing.assert_allclose(result.to_numpy(), a, rtol=1e-5)
        assert [list(s.shape) for s in result.local_shards] == [
            list(t.shape) for t in result.layout.local_types
        ]

    # ─── transfer_to: the collective each change runs ───────────────

    @pytest.mark.parametrize(
        ("rows", "source", "target", "expected"),
        [
            (8, (Sharded(0),), (Sharded(0),), []),
            (8, (Partial(),), (Partial(),), []),
            (8, (Partial(),), (Replicated(),), ["allreduce_sum"]),
            (8, (Partial(),), (Sharded(0),), ["reduce_scatter"]),
            (7, (Partial(),), (Sharded(0),), ["reduce_scatter"]),
            (8, (Sharded(0),), (Replicated(),), ["allgather"]),
            (8, (Sharded(0),), (Sharded(1),), ["all_to_all"]),
            (7, (Sharded(0),), (Sharded(1),), ["allgather"]),
            (8, (Replicated(),), (Sharded(1),), []),
            (8, (Replicated(),), (Partial(),), []),
        ],
        ids=[
            "S0->S0",
            "P->P",
            "P->R",
            "P->S0",
            "P->S0-uneven",
            "S0->R",
            "S0->S1",
            "S0->S1-uneven",
            "R->S1",
            "R->P",
        ],
    )
    def test_transfer_runs_the_collective_of_the_change(
        self,
        rows: int,
        source: tuple[Placement, ...],
        target: tuple[Placement, ...],
        expected: list[str],
    ) -> None:
        """7 rows over 4 devices are uneven shards, the larger ones first,
        so swapping their split axis gathers instead of an all_to_all."""
        a = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8)
        mesh = self.MESH_1D
        if source == (Partial(),):
            t = self.partial_fn(a, mesh, source)
            full = a * mesh.num_devices
        else:
            t = transfer_to(Tensor(a), DeviceMapping(mesh, source))
            full = a
        with _collectives_called() as names:
            result = transfer_to(t, DeviceMapping(mesh, target))
        assert names == expected
        assert result.placements == target
        if target == (Partial(),):
            # Replicated to Partial keeps one copy, so the sum is the input.
            np.testing.assert_allclose(result.to_numpy(), full, rtol=1e-5)
            return
        for i, shard in enumerate(result.local_shards):
            np.testing.assert_allclose(
                shard.to_numpy(),
                self._expected_shard(full, i, mesh, target),
                rtol=1e-5,
            )

    def test_transfer_symbolic_rows_in_one_graph(self) -> None:
        """A symbolic row count may not split evenly, so swapping the split
        axis gathers and slices instead of running an all-to-all."""
        mesh = self.MESH_1D
        sharded, partial = (
            TensorLayout(
                DType.float32, ["rows", 8], DeviceMapping(mesh, (placement,))
            )
            for placement in (Sharded(0), Partial())
        )
        with _collectives_called() as names:
            staged = _SymbolicTransfers().trace(sharded, partial)
        assert sorted(names) == [
            "allgather",
            "allgather",
            "allreduce_sum",
            "reduce_scatter",
        ]
        assert [layout.placements for layout in staged.out_layouts] == [
            (Replicated(),),
            (Sharded(1),),
            (Replicated(),),
            (Sharded(0),),
        ]
        for layout in staged.out_layouts:
            rows, columns = layout.shape
            assert "rows" in str(rows) and int(columns) == 8
