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
"""IR-shape tests for ``DEFAULT_TRANSITIONS`` versus ``ALL_TRANSITIONS``.

With the default transitions the picker resolves every ``Partial`` to
``Replicated`` (pure tensor-parallel):
rms_norm runs on the full ``[batch, 8]`` tensor and the residual stream
never shards. ``ALL_TRANSITIONS`` adds ``(Partial, Sharded)`` and lets the
picker take the locally cheaper reduce-scatter, which on this
``matmul -> matmul -> norm -> residual_add`` topology splits the
residual stream along the batch axis (the algebraic dim
``div(batch, 2)``) and rejoins it, the sequence-parallel roundtrip.

These tests run on a 2-CPU simulated mesh, where the collective lowering
inlines reductions as direct per-device ops (no literal ``allreduce`` /
``allgather`` strings appear in the IR). The presence or absence of
``div(batch`` / ``rmo.mo.slice`` on the residual stream is the
load-bearing binary signal: the algebraic split only appears when the
picker chose the sequence-parallel layout.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental.functional import (
    add,
    matmul,
    relu,
    rms_norm,
    transfer_to,
)
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.sharding import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Transition,
    auto_reshard,
)
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor

F32 = DType.float32
HIDDEN = 8
FFL = 16
BATCH = 3


def _mesh_2() -> DeviceMesh:
    return DeviceMesh(
        devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",)
    )


@module_dataclass
class TPBlock(Module[[Tensor], Tensor]):
    """A minimal transformer-block-shaped subgraph.

    ``W_up`` is column-parallel (Sharded on the N axis), ``W_down`` is
    row-parallel (Sharded on the K axis). Their composition produces a
    ``Partial`` activation that must then feed a replicated rms_norm and
    a residual add — the exact topology where Gemma3 hit the SP
    roundtrip.
    """

    W_up: Tensor
    W_down: Tensor
    norm_w: Tensor

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = matmul(x, self.W_up)  # Sharded(N)
        h = relu(h)
        h = matmul(h, self.W_down)  # Partial(SUM)
        h = rms_norm(h, self.norm_w, 1e-6)
        return add(residual, h)


@module_dataclass
class MatMulReluMatMul(Module[[Tensor], Tensor]):
    """The README's SP worked example: ``matmul -> relu -> matmul``."""

    W_up: Tensor
    W_down: Tensor

    def forward(self, x: Tensor) -> Tensor:
        h = matmul(x, self.W_up)  # Sharded(N)
        h = relu(h)
        return matmul(h, self.W_down)  # Partial -> graph output (unconsumed)


def _make_tp_block(mesh: DeviceMesh) -> TPBlock:
    rng = np.random.default_rng(0)
    W_up = rng.standard_normal((HIDDEN, FFL)).astype(np.float32) * 0.1
    W_down = rng.standard_normal((FFL, HIDDEN)).astype(np.float32) * 0.1
    norm_w = np.ones((HIDDEN,), dtype=np.float32)
    return TPBlock(
        W_up=transfer_to(Tensor(W_up), DeviceMapping(mesh, (Sharded(1),))),
        W_down=transfer_to(Tensor(W_down), DeviceMapping(mesh, (Sharded(0),))),
        norm_w=transfer_to(
            Tensor(norm_w), DeviceMapping(mesh, (Replicated(),))
        ),
    )


def _make_matmul_relu_matmul(mesh: DeviceMesh) -> MatMulReluMatMul:
    rng = np.random.default_rng(1)
    W_up = rng.standard_normal((HIDDEN, FFL)).astype(np.float32) * 0.1
    W_down = rng.standard_normal((FFL, HIDDEN)).astype(np.float32) * 0.1
    return MatMulReluMatMul(
        W_up=transfer_to(Tensor(W_up), DeviceMapping(mesh, (Sharded(1),))),
        W_down=transfer_to(Tensor(W_down), DeviceMapping(mesh, (Sharded(0),))),
    )


PURE_TP = DEFAULT_TRANSITIONS
SEQUENCE_PARALLEL = ALL_TRANSITIONS


def _block_ir(allow: frozenset[Transition]) -> str:
    mesh = _mesh_2()
    with auto_reshard(allow, mode="silent"):
        block = _make_tp_block(mesh)
        input_type = TensorLayout(
            F32, ["batch", HIDDEN], DeviceMapping(mesh, (Replicated(),))
        )
        return repr(block.trace(input_type))


def _matmul_relu_matmul_ir(allow: frozenset[Transition]) -> str:
    mesh = _mesh_2()
    with auto_reshard(allow, mode="silent"):
        block = _make_matmul_relu_matmul(mesh)
        input_type = TensorLayout(
            F32, ["batch", HIDDEN], DeviceMapping(mesh, (Replicated(),))
        )
        return repr(block.trace(input_type))


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestReduceScatterPermission:
    """Pins the picker's behavior on the Gemma3-shaped block topology."""

    def test_without_reduce_scatter_is_pure_tp(self) -> None:
        """Without ``(Partial, Sharded)`` the Partial resolves to Replicated.

        rms_norm runs on the full ``[batch, 8]`` tensor; the residual
        stream stays Replicated; no algebraic per-rank drift, no slices.
        """
        ir = _block_ir(PURE_TP)
        assert "div(batch" not in ir, (
            "Without (Partial, Sharded) the residual stream must stay "
            "Replicated; the algebraic per-rank dim ``div(batch, 2)`` "
            "should not appear."
        )
        assert "rmo.mo.slice" not in ir, (
            "Without (Partial, Sharded) no per-rank slice ops may appear "
            "on the residual stream."
        )

    def test_with_reduce_scatter_enables_sp_roundtrip(self) -> None:
        """Permitting ``(Partial, Sharded)`` lets the picker take SP.

        The residual stream is split via ``rmo.mo.slice`` carrying the
        algebraic per-rank dim ``div(batch, 2)``.
        """
        ir = _block_ir(SEQUENCE_PARALLEL)
        assert "div(batch" in ir, (
            "Permitting (Partial, Sharded) should let the picker shard "
            "the residual stream, producing ``div(batch, 2)``."
        )
        assert "rmo.mo.slice" in ir, (
            "Permitting (Partial, Sharded) should emit per-rank slice "
            "ops for the SP path."
        )

    def test_matmul_relu_matmul_straightline_by_default(self) -> None:
        """SP-friendly chains need no resolution either way.

        On ``matmul -> relu -> matmul`` the intermediate is ``Sharded``
        (not ``Partial``) and the trailing ``Partial`` is the unconsumed
        graph output, so neither setting inserts a slice. This documents
        that the permission only acts where a ``Partial`` is consumed.
        """
        assert "rmo.mo.slice" not in _matmul_relu_matmul_ir(PURE_TP)
        assert "rmo.mo.slice" not in _matmul_relu_matmul_ir(SEQUENCE_PARALLEL)
