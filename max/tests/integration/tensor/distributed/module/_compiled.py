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
"""Shared test logic for compiled execution with distributed weights.

DO NOT run this file directly -- it contains base classes that are
subclassed by module/test_compiled_simulated.py and
module/test_compiled_multi_gpu.py.

Subclasses must define:
    MESH_2D: DeviceMesh  -- 4 devices, shape (2,2), axis_names=("dp","tp")
    MESH_2: DeviceMesh   -- 2 devices, shape (2,), axis_names=("tp",)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.experimental.distributed_functional import (
    full,
    matmul,
    ones,
    relu,
    transfer_to,
)
from max.experimental.nn.module import CompiledModel, Module, module_dataclass
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor, TensorType

D = 4
F32 = DType.float32


def sym(feature_dim: int) -> TensorType:
    """Symbolic input type -- always CPU regardless of mesh."""
    return TensorType(F32, ["batch", feature_dim], device=CPU())


def check(result: Tensor, expected_val: float, shape: tuple[int, ...]) -> None:
    """Materialize and check values match expected constant."""
    gathered = result.materialize() if result.is_distributed else result
    expected = np.full(shape, expected_val)
    np.testing.assert_allclose(gathered.to_numpy(), expected, rtol=1e-4)


class CompiledTests:
    """Base class with all 6 compiled test methods.

    Subclass must set MESH_2D and MESH_2.
    """

    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]

    def test_col_tp_linear(self) -> None:
        mesh = self.MESH_2

        @module_dataclass
        class Linear(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(x_dist, self.W)

        W = ones(
            [D, D], dtype=F32, device=PlacementMapping(mesh, (Sharded(1),))
        )
        compiled = Linear(W=W).compile(sym(D))
        result = compiled(full([3, D], 1.0, dtype=F32, device=CPU()))
        assert result.placements == (Sharded(1),)
        check(result, float(D), (3, D))

    def test_tp_mlp(self) -> None:
        mesh = self.MESH_2

        @module_dataclass
        class MLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(relu(matmul(x_dist, self.W1)), self.W2)

        m_s1 = PlacementMapping(mesh, (Sharded(1),))
        m_s0 = PlacementMapping(mesh, (Sharded(0),))
        model = MLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        compiled = model.compile(sym(D))
        check(
            compiled(full([4, D], 1.0, dtype=F32, device=CPU())),
            float(D * D),
            (4, D),
        )

    def test_dynamic_batch(self) -> None:
        mesh = self.MESH_2

        @module_dataclass
        class MLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(relu(matmul(x_dist, self.W1)), self.W2)

        m_s1 = PlacementMapping(mesh, (Sharded(1),))
        m_s0 = PlacementMapping(mesh, (Sharded(0),))
        model = MLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        compiled = model.compile(sym(D))
        for batch in [3, 7]:
            check(
                compiled(full([batch, D], 1.0, dtype=F32, device=CPU())),
                float(D * D),
                (batch, D),
            )

    def test_eager_vs_compiled(self) -> None:
        mesh = self.MESH_2

        @module_dataclass
        class MLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(relu(matmul(x_dist, self.W1)), self.W2)

        m_s1 = PlacementMapping(mesh, (Sharded(1),))
        m_s0 = PlacementMapping(mesh, (Sharded(0),))
        model = MLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        x = full([5, D], 1.0, dtype=F32, device=CPU())
        eager_result = model(x)
        assert eager_result.placements == (Partial(),)
        eager = eager_result.to_numpy()
        compiled_result = model.compile(sym(D))(x)
        assert compiled_result.placements == (Partial(),)
        compiled = compiled_result.to_numpy()
        np.testing.assert_allclose(eager, compiled, rtol=1e-4)

    def test_compiled_with_allreduce(self) -> None:
        mesh = self.MESH_2

        @module_dataclass
        class TPMLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                h = relu(matmul(x_dist, self.W1))
                out = matmul(h, self.W2)
                return transfer_to(out, PlacementMapping(mesh, (Replicated(),)))

        m_s1 = PlacementMapping(mesh, (Sharded(1),))
        m_s0 = PlacementMapping(mesh, (Sharded(0),))
        model = TPMLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        compiled = model.compile(sym(D))
        result = compiled(full([3, D], 1.0, dtype=F32, device=CPU()))
        assert result.placements == (Replicated(),)
        check(result, float(D * D), (3, D))

    def test_2d_mesh(self) -> None:
        mesh = self.MESH_2D

        @module_dataclass
        class MLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(
                    x, PlacementMapping(mesh, (Replicated(), Replicated()))
                )
                return matmul(relu(matmul(x_dist, self.W1)), self.W2)

        m_s1 = PlacementMapping(mesh, (Replicated(), Sharded(1)))
        m_s0 = PlacementMapping(mesh, (Replicated(), Sharded(0)))
        model = MLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        compiled = model.compile(sym(D))
        result = compiled(full([3, D], 1.0, dtype=F32, device=CPU()))
        assert result.is_distributed
        assert result.placements == (Replicated(), Partial())
        check(result, float(D * D), (3, D))

    # ── CompiledModel API tests ─────────────────────────────────────────

    def test_compile_returns_compiled_model(self) -> None:
        """compile() returns a CompiledModel, not a bare closure."""
        mesh = self.MESH_2

        @module_dataclass
        class Linear(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(x_dist, self.W)

        W = ones(
            [D, D], dtype=F32, device=PlacementMapping(mesh, (Sharded(1),))
        )
        compiled = Linear(W=W).compile(sym(D))
        assert isinstance(compiled, CompiledModel)
        assert isinstance(compiled.engine_model, Model)
        assert isinstance(compiled.signal_buffers, list)

    def test_execute_raw(self) -> None:
        """execute_raw() returns flat Buffers without Tensor wrapping."""
        mesh = self.MESH_2

        @module_dataclass
        class Linear(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                return matmul(x_dist, self.W)

        W = ones(
            [D, D], dtype=F32, device=PlacementMapping(mesh, (Sharded(1),))
        )
        compiled = Linear(W=W).compile(sym(D))
        x = Buffer.from_numpy(np.ones((3, D), dtype=np.float32))
        raw_outputs = compiled.execute_raw(x)

        assert isinstance(raw_outputs, list)
        assert all(isinstance(buf, Buffer) for buf in raw_outputs)
        # Distributed model with 2 devices should return 2 output shards.
        assert len(raw_outputs) == mesh.num_devices

    def test_execute_raw_matches_call(self) -> None:
        """execute_raw() produces the same values as __call__()."""
        mesh = self.MESH_2

        @module_dataclass
        class MLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dist = transfer_to(x, PlacementMapping(mesh, (Replicated(),)))
                h = relu(matmul(x_dist, self.W1))
                out = matmul(h, self.W2)
                return transfer_to(out, PlacementMapping(mesh, (Replicated(),)))

        m_s1 = PlacementMapping(mesh, (Sharded(1),))
        m_s0 = PlacementMapping(mesh, (Sharded(0),))
        model = MLP(
            W1=ones([D, D], dtype=F32, device=m_s1),
            W2=ones([D, D], dtype=F32, device=m_s0),
        )
        compiled = model.compile(sym(D))

        x_np = np.ones((3, D), dtype=np.float32)
        # Tensor path
        tensor_result = compiled(full([3, D], 1.0, dtype=F32, device=CPU()))
        tensor_np = tensor_result.to_numpy()
        # Buffer path
        raw_outputs = compiled.execute_raw(Buffer.from_numpy(x_np))
        raw_np = np.from_dlpack(raw_outputs[0].to(CPU()))

        np.testing.assert_allclose(raw_np, tensor_np, rtol=1e-4)
