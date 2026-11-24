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
"""Collective ops tests on simulated CPU mesh — inherits all logic from _collectives."""

from unittest.mock import patch

import numpy as np
import pytest
from _test_helpers import make_partial
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.tensor import Tensor
from max.tensor.realization_context import EagerRealizationContext
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    TensorLayout,
)

from functional._collectives import CollectivesTests


class TestCollectives(CollectivesTests):
    MESH_1D = DeviceMesh(
        devices=(CPU(), CPU(), CPU(), CPU()),
        mesh_shape=(4,),
        axis_names=("tp",),
    )
    MESH_2D = DeviceMesh(
        devices=(CPU(), CPU(), CPU(), CPU()),
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
    )
    partial_fn = staticmethod(make_partial)


class _Exchange(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Runs the Ulysses exchange and its inverse on a row-sharded tensor."""

    def forward(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        swapped = F.all_to_all(rows, split_axis=1, concat_axis=0)
        # Every rank now holds every row and a quarter of the columns, so the
        # row axis is whole on each device rather than carried as an
        # expression over the shards.
        swapped = F.rebind(swapped, [Dim("rows"), rows.shape[1]])
        back = F.all_to_all(swapped, split_axis=0, concat_axis=1)
        return swapped, F.rebind(back, [Dim("rows"), rows.shape[1]])


def test_all_to_all_exchange_compiles_with_a_symbolic_dim() -> None:
    """The one-device exchange must compile with a symbolic shard dim.

    A mesh that repeats one CPU runs no kernel: each shard is split and the
    received chunks concatenated on that device. Splitting the whole
    concatenation instead leaves an algebraically sized concat whose split
    the compiler rejected at four ranks with "statically known dimensions
    must match", so both directions of the exchange are built here with
    symbolic rows and checked for the values an all-to-all must give.
    """
    mesh = TestCollectives.MESH_1D
    layout = TensorLayout(
        DType.float32, ["rows", 8], DeviceMapping(mesh, (Sharded(0),))
    )
    a = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    compiled = _Exchange().compile(layout)
    swapped, back = compiled(
        F.transfer_to(Tensor(a), DeviceMapping(mesh, (Sharded(0),)))
    )
    # Rank ``j`` holds every row and the columns each rank sent it.
    for rank, shard in enumerate(swapped.local_shards):
        np.testing.assert_allclose(
            shard.to_numpy(), a[:, 2 * rank : 2 * (rank + 1)]
        )
    np.testing.assert_allclose(back.to_numpy(), a)


def test_groups_on_one_device_allocate_no_signal_buffers() -> None:
    """Every device of this mesh is the same CPU, so no group exchanges
    anything and eager execution creates no signal buffers."""
    mesh = TestCollectives.MESH_2D
    t = make_partial(
        np.ones((2, 4), dtype=np.float32), mesh, (Replicated(), Partial())
    )
    with patch.object(
        EagerRealizationContext, "ensure_signal_buffers"
    ) as ensure:
        F.allreduce_sum(t, mesh_axis="tp").to_numpy()
    ensure.assert_not_called()


def test_post_norm_residual_rms_norm_requires_replicated_operands() -> None:
    """Each device joins its own operands under a replicated label.

    A residual or weight that differs per device would make the outputs
    differ under that label, so it is rejected; a host weight passes the
    check, to be copied to every device. The CPU mesh then has no kernel to
    run, which is the error a host weight reaches.
    """
    mesh = TestCollectives.MESH_1D
    replicated = DeviceMapping(mesh, (Replicated(),))
    with F.lazy():
        t = Tensor.zeros(
            [2, 8], dtype=DType.bfloat16, device=replicated
        ).rebind_mapping(DeviceMapping(mesh, (Partial(),)))
        residual = Tensor.zeros([2, 8], dtype=DType.bfloat16, device=replicated)
        gamma = Tensor.zeros([8], dtype=DType.bfloat16, device=replicated)
        mix = Tensor.zeros([2], dtype=DType.float32, device=replicated)

        with pytest.raises(ValueError, match="residual replicated"):
            F.allreduce_post_norm_residual_rms_norm(
                t,
                residual.rebind_mapping(DeviceMapping(mesh, (Partial(),))),
                gamma,
                gamma,
                mix,
                1e-6,
            )
        with pytest.raises(ValueError, match="gamma2 replicated"):
            F.allreduce_post_norm_residual_rms_norm(
                t,
                residual,
                gamma,
                gamma.rebind_mapping(DeviceMapping(mesh, (Sharded(0),))),
                mix,
                1e-6,
            )
        host_mix = Tensor.zeros([2], dtype=DType.float32, device=CPU())
        with pytest.raises(ValueError, match="only on accelerators"):
            F.allreduce_post_norm_residual_rms_norm(
                t, residual, gamma, gamma, host_mix, 1e-6
            )


def test_exit_concat_rejects_indivisible_cols() -> None:
    """The no-signals fallback keeps the op's equal-shard requirement.

    The kernel path rejects columns that do not divide by the device count;
    the fallback must not silently drop the remainder columns instead.
    """
    mesh = TestCollectives.MESH_1D
    replicated = DeviceMapping(mesh, (Replicated(),))
    with F.lazy():
        t = Tensor.zeros(
            [2, 10], dtype=DType.float32, device=replicated
        ).rebind_mapping(DeviceMapping(mesh, (Partial(),)))
        gate_up = Tensor.zeros([2, 8], dtype=DType.bfloat16, device=replicated)
        gamma = Tensor.zeros([10], dtype=DType.float32, device=replicated)

        with pytest.raises(ValueError, match="multiple of the device count"):
            F.allreduce_rms_norm_exit_concat(
                t, gate_up, gamma, epsilon=1e-5, beta=1.5, linear_beta=1.5
            )
