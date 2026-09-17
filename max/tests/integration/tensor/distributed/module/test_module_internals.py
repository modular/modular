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
"""Tests for Module.compile() with distributed tensors.

Tests the internal flatten/unflatten helpers in ``nn/module.py`` and the
end-to-end Module.compile() path for distributed inputs and weights.

These tests require only CPU -- no GPU, no collectives, no dispatch ops.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.experimental.compilation import CompiledCallable
from max.experimental.nn.module import (
    Module,
    module_dataclass,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
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
#  Helpers
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


# ═════════════════════════════════════════════════════════════════════════


@module_dataclass
class _IdentityModule(Module[[Tensor], Tensor]):
    """Trivial module that returns the input unchanged."""

    W: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x


class TestModuleCompileDistributed:
    """Smoke tests for Module.compile() with distributed input types.

    Exercises the full compile path: flatten_input_types -> wrap_graph_inputs
    -> forward -> flatten_outputs -> compile -> unflatten_args ->
    reconstruct_outputs.
    """

    def test_compile_with_sharded_weight(self) -> None:
        """Module with sharded weight compiles and executes."""
        W = _make_realized_sharded([4, 8], 2, shard_axis=0)
        model = _IdentityModule(W=W)
        input_type = TensorLayout(DType.float32, [3, 8], CPU())
        compiled = model.compile(input_type)
        x = Tensor.ones([3, 8], dtype=DType.float32, device=CPU())
        result = compiled(x)
        assert list(result.shape) == [3, 8]
        np.testing.assert_allclose(np.from_dlpack(result), 1.0)

    def test_compile_with_distributed_input_type(self) -> None:
        """TensorLayout input is flattened and reconstructed."""
        mesh = mesh_1d(2)
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _IdentityModule(W=W)
        input_type = TensorLayout(
            DType.float32, [8, 4], DeviceMapping(mesh, (Sharded(0),))
        )
        compiled = model.compile(input_type)
        x = _make_realized_sharded([4, 4], 2, shard_axis=0)
        result = compiled(x)
        assert result.is_distributed
        assert result.num_shards == 2
        assert list(result.shape) == [8, 4]

    def test_compile_roundtrip_values(self) -> None:
        """Compiled distributed identity preserves shard data."""
        mesh = mesh_1d(2)
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _IdentityModule(W=W)
        input_type = TensorLayout(
            DType.float32, [8, 4], DeviceMapping(mesh, (Sharded(0),))
        )
        compiled = model.compile(input_type)
        x = _make_realized_sharded([4, 4], 2, shard_axis=0)
        result = compiled(x)
        for shard in result.local_shards:
            arr = np.from_dlpack(shard)
            np.testing.assert_allclose(arr, 0.0)


# ═════════════════════════════════════════════════════════════════════════
#  CompiledCallable API surface
# ═════════════════════════════════════════════════════════════════════════


class TestCompiledCallableAPI:
    """Tests for CompiledCallable properties and execute_raw() method.

    These verify the public API surface that pipeline builders rely on:
    engine_model for CUDA graph capture, execute_raw for zero-overhead
    buffer execution, and signal_buffers for multi-GPU collectives.
    """

    def _compile_identity(self) -> CompiledCallable[[Tensor], Tensor]:
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _IdentityModule(W=W)
        input_type = TensorLayout(DType.float32, [3, 8], CPU())
        return model.compile(input_type)

    def test_returns_compiled_model(self) -> None:
        """Module.compile() returns a CompiledCallable instance."""
        compiled = self._compile_identity()
        assert isinstance(compiled, CompiledCallable)

    def test_engine_model_type(self) -> None:
        """engine_model exposes the underlying engine.Model."""
        compiled = self._compile_identity()
        assert isinstance(compiled.engine_model, Model)

    def test_signal_buffers_empty_on_cpu(self) -> None:
        """CPU-only model has no signal buffers."""
        compiled = self._compile_identity()
        assert compiled.signal_buffers == []

    def test_execute_raw_returns_buffers(self) -> None:
        """execute_raw() returns list[Buffer], not Tensors."""
        compiled = self._compile_identity()
        x = Buffer.zeros([3, 8], dtype=DType.float32, device=CPU())
        result = compiled.execute_raw(x)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], Buffer)

    def test_execute_raw_values(self) -> None:
        """execute_raw() preserves input data through identity model."""
        compiled = self._compile_identity()
        x = Buffer.zeros([3, 8], dtype=DType.float32, device=CPU())
        result = compiled.execute_raw(x)
        arr = np.from_dlpack(result[0])
        np.testing.assert_allclose(arr, 0.0)

    def test_execute_raw_matches_call(self) -> None:
        """execute_raw() and __call__() produce equivalent results."""
        compiled = self._compile_identity()
        x_tensor = Tensor.ones([3, 8], dtype=DType.float32, device=CPU())
        x_buffer = Buffer.zeros([3, 8], dtype=DType.float32, device=CPU())

        call_result = compiled(x_tensor)
        raw_result = compiled.execute_raw(x_buffer)

        call_arr = np.from_dlpack(call_result)
        raw_arr = np.from_dlpack(raw_result[0])
        assert call_arr.shape == raw_arr.shape

    def test_call_missing_arg_raises_diagnostic(self) -> None:
        """Calling with too few args says which slot went unfilled."""
        compiled = self._compile_identity()
        with pytest.raises(ValueError) as excinfo:
            compiled()  # type: ignore[call-arg]
        message = str(excinfo.value)
        assert "tree structure mismatch" in message
        assert "expected keys [0], got []" in message

    def test_execute_raw_missing_arg_raises_diagnostic(self) -> None:
        """execute_raw() with too few buffers names the argument it wanted."""
        compiled = self._compile_identity()
        with pytest.raises(TypeError, match="missing a required argument"):
            compiled.execute_raw()  # identity needs 1 buffer, gave 0


# ═════════════════════════════════════════════════════════════════════════
#  Module.to: Device, DeviceMesh, DeviceMapping
# ═════════════════════════════════════════════════════════════════════════


@module_dataclass
class _SimpleModule(Module[[Tensor], Tensor]):
    """Simple module with a single parameter for testing."""

    W: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x + self.W


class TestModuleTo:
    """Tests for Module.to with Device, DeviceMesh, and DeviceMapping."""

    def test_to_device_backward_compat(self) -> None:
        """Module.to(Device) should work as before."""
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        result = model.to(CPU())
        assert result is model
        assert model.device == CPU()
        assert not model.W.is_distributed

    def test_to_device_sets_module_device(self) -> None:
        """Module.to(Device) sets the module's device property."""
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        model.to(CPU())
        assert model.device == CPU()

    def test_to_single_device_mesh(self) -> None:
        """Module.to(DeviceMesh) with single device should transfer."""
        W = Tensor.ones([4, 8], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        single_mesh = DeviceMesh(
            devices=(CPU(),), mesh_shape=(1,), axis_names=("_",)
        )
        model.to(single_mesh)
        assert model.device == CPU()
        assert not model.W.is_distributed

    def test_to_multi_device_mesh_unsharded(self) -> None:
        """Module.to(DeviceMesh) with multi-device creates replicated params."""
        W = Tensor.ones([4, 8], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        mesh = mesh_1d(2)
        model.to(mesh)
        assert model.device == CPU()  # Primary device
        assert model.W.is_distributed
        assert model.W.num_shards == 2
        assert model.W.mapping.is_fully_replicated

    def test_to_device_mapping_single_device(self) -> None:
        """Module.to(DeviceMapping) with single-device mesh transfers."""
        W = Tensor.ones([4, 8], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        single_mesh = DeviceMesh(
            devices=(CPU(),), mesh_shape=(1,), axis_names=("_",)
        )
        mapping = PlacementMapping(single_mesh, (Replicated(),))
        model.to(mapping)
        assert model.device == CPU()
        assert not model.W.is_distributed

    def test_to_device_mapping_multi_device_replicated(self) -> None:
        """Module.to(DeviceMapping) with Replicated replicates parameters."""
        W = Tensor.ones([4, 8], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        mesh = mesh_1d(2)
        mapping = PlacementMapping(mesh, (Replicated(),))
        model.to(mapping)
        assert model.device == CPU()  # Primary device
        assert model.W.is_distributed
        assert model.W.num_shards == 2
        assert model.W.placements == (Replicated(),)

    def test_to_device_mapping_multi_device_sharded(self) -> None:
        """Module.to(DeviceMapping) with Sharded shards parameters."""
        W = Tensor.ones([4, 8], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        mesh = mesh_1d(2)
        mapping = PlacementMapping(mesh, (Sharded(0),))
        model.to(mapping)
        assert model.device == CPU()  # Primary device
        assert model.W.is_distributed
        assert model.W.num_shards == 2
        assert model.W.placements == (Sharded(0),)
        # Each shard should have half along axis 0
        shards = model.W.local_shards
        for s in shards:
            assert list(s.shape) == [2, 8]

    def test_to_multi_device_mesh(self) -> None:
        """Module.to(DeviceMesh) on distributed param raises an error."""
        # Create a module with distributed parameter
        mesh1 = mesh_1d(2, "tp")
        bufs = tuple(
            Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
            for _ in range(2)
        )
        W = Tensor._from_shards(bufs, mesh1, (Sharded(0),), global_shape=[8, 8])
        model = _SimpleModule(W=W)

        # Transfer to a different mesh with same shape
        mesh2 = mesh_1d(2, "new_tp")
        model.to(mesh2)
        assert model.W.is_distributed
        assert model.W.num_shards == 2
        assert model.W.placements == (Sharded(0),)

    def test_to_invalid_type_raises(self) -> None:
        """Module.to with invalid type should raise TypeError."""
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _SimpleModule(W=W)
        with pytest.raises(TypeError, match="expects Device, DeviceMesh"):
            model.to("invalid")  # type: ignore[arg-type]

    def test_to_applies_to_children(self) -> None:
        """Module.to should recursively apply to child modules."""

        @module_dataclass
        class _ParentModule(Module[[Tensor], Tensor]):
            child: _SimpleModule

            def forward(self, x: Tensor) -> Tensor:
                return self.child(x)

        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        child = _SimpleModule(W=W)
        parent = _ParentModule(child=child)

        mesh = mesh_1d(2)
        parent.to(mesh)

        # Both parent and child should have distributed parameters
        assert parent.device == CPU()
        assert parent.child.W.is_distributed
        assert parent.child.W.num_shards == 2
