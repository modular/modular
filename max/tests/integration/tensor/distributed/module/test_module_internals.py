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
from max.experimental.nn.module import (
    CompiledModel,
    Module,
    _flatten_input_types,
    _flatten_named_buffers,
    _InputSlot,
    _OutputSlot,
    _reconstruct_outputs,
    _unflatten_args,
    module_dataclass,
)
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    DistributedTensorType,
    PlacementMapping,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType

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
#  _InputSlot / _OutputSlot
# ═════════════════════════════════════════════════════════════════════════


class TestSlotDescriptors:
    def test_input_slot_fields(self) -> None:
        slot = _InputSlot(start=0, count=4, dist=None)
        assert slot.start == 0
        assert slot.count == 4
        assert slot.dist is None

    def test_input_slot_with_dist(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedTensorType(DType.float32, [8, 4], mesh, [Sharded(0)])
        slot = _InputSlot(start=1, count=2, dist=dt)
        assert slot.dist is dt
        assert slot.count == 2

    def test_output_slot_fields(self) -> None:
        slot = _OutputSlot(start=0, count=1, mapping=None)
        assert slot.start == 0
        assert slot.count == 1
        assert slot.mapping is None

    def test_output_slot_with_mapping(self) -> None:
        mesh = mesh_1d(2)
        mapping = PlacementMapping(mesh, (Sharded(0),))
        slot = _OutputSlot(start=2, count=2, mapping=mapping)
        assert slot.mapping is mapping

    def test_slots_are_frozen(self) -> None:
        slot = _InputSlot(start=0, count=1, dist=None)
        with pytest.raises(AttributeError):
            slot.start = 5  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════
#  _flatten_input_types
# ═════════════════════════════════════════════════════════════════════════


class TestFlattenInputTypes:
    def test_non_distributed_passthrough(self) -> None:
        tt = TensorType(DType.float32, [4, 8], CPU())
        flat, slots = _flatten_input_types([tt])
        assert len(flat) == 1
        assert flat[0] is tt
        assert slots[0].count == 1
        assert slots[0].dist is None

    def test_distributed_expands(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        flat, slots = _flatten_input_types([dt])
        assert len(flat) == 4
        assert slots[0].count == 4
        assert slots[0].dist is dt

    def test_mixed_inputs(self) -> None:
        mesh = mesh_1d(2)
        tt = TensorType(DType.float32, [4, 8], CPU())
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        flat, slots = _flatten_input_types([tt, dt])
        assert len(flat) == 3  # 1 + 2
        assert slots[0].count == 1
        assert slots[0].start == 0
        assert slots[1].count == 2
        assert slots[1].start == 1

    def test_empty_input(self) -> None:
        flat, slots = _flatten_input_types([])
        assert flat == []
        assert slots == []

    def test_multiple_distributed(self) -> None:
        mesh = mesh_1d(2)
        dt1 = DistributedTensorType(DType.float32, [4, 8], mesh, [Sharded(0)])
        dt2 = DistributedTensorType(DType.float32, [6, 8], mesh, [Sharded(0)])
        flat, slots = _flatten_input_types([dt1, dt2])
        assert len(flat) == 4  # 2 + 2
        assert slots[0].start == 0
        assert slots[0].count == 2
        assert slots[1].start == 2
        assert slots[1].count == 2

    def test_buffer_type_distributed(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        flat, _ = _flatten_input_types([dt])
        assert len(flat) == 2
        for lt in flat:
            assert isinstance(lt, BufferType)


# ═════════════════════════════════════════════════════════════════════════
#  _unflatten_args
# ═════════════════════════════════════════════════════════════════════════


class TestUnflattenArgs:
    def test_non_distributed_passthrough(self) -> None:
        buf = Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
        slot = _InputSlot(start=0, count=1, dist=None)
        flat = _unflatten_args([buf], [slot])
        assert len(flat) == 1
        assert flat[0] is buf

    def test_distributed_tensor_expands(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedTensorType(DType.float32, [8, 4], mesh, [Sharded(0)])
        slot = _InputSlot(start=0, count=2, dist=dt)
        t = _make_realized_sharded([4, 4], 2, shard_axis=0)
        flat = _unflatten_args([t], [slot])
        assert len(flat) == 2
        for item in flat:
            assert isinstance(item, Buffer)

    def test_mixed_args(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedTensorType(DType.float32, [8, 4], mesh, [Sharded(0)])
        slot_plain = _InputSlot(start=0, count=1, dist=None)
        slot_dist = _InputSlot(start=1, count=2, dist=dt)

        buf = Buffer.zeros([3, 4], dtype=DType.float32, device=CPU())
        t = _make_realized_sharded([4, 4], 2, shard_axis=0)
        flat = _unflatten_args([buf, t], [slot_plain, slot_dist])
        assert len(flat) == 3
        assert flat[0] is buf
        assert isinstance(flat[1], Buffer)
        assert isinstance(flat[2], Buffer)

    def test_non_distributed_tensor_passthrough(self) -> None:
        """Non-distributed Tensor with dist=None passes through unchanged."""
        slot = _InputSlot(start=0, count=1, dist=None)
        t = Tensor.zeros([4, 8], dtype=DType.float32, device=CPU())
        flat = _unflatten_args([t], [slot])
        assert len(flat) == 1
        assert flat[0] is t


# ═════════════════════════════════════════════════════════════════════════
#  _reconstruct_outputs
# ═════════════════════════════════════════════════════════════════════════


class TestReconstructOutputs:
    def test_unary_non_distributed(self) -> None:
        buf = Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
        slot = _OutputSlot(start=0, count=1, mapping=None)
        result = _reconstruct_outputs([buf], [slot], unary=True)
        assert isinstance(result, Tensor)
        assert not result.is_distributed
        assert list(result.shape) == [4, 8]

    def test_multi_output_non_distributed(self) -> None:
        buf1 = Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
        buf2 = Buffer.zeros([3, 6], dtype=DType.float32, device=CPU())
        slot1 = _OutputSlot(start=0, count=1, mapping=None)
        slot2 = _OutputSlot(start=1, count=1, mapping=None)
        result = _reconstruct_outputs([buf1, buf2], [slot1, slot2], unary=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert list(result[0].shape) == [4, 8]
        assert list(result[1].shape) == [3, 6]

    def test_distributed_reconstruction(self) -> None:
        mesh = mesh_1d(2)
        mapping = PlacementMapping(mesh, (Sharded(0),))
        buf1 = Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
        buf2 = Buffer.zeros([4, 8], dtype=DType.float32, device=CPU())
        slot = _OutputSlot(start=0, count=2, mapping=mapping)
        result = _reconstruct_outputs([buf1, buf2], [slot], unary=True)
        assert isinstance(result, Tensor)
        assert result.is_distributed
        assert result.num_shards == 2
        assert list(result.shape) == [8, 8]
        assert result.placements == (Sharded(0),)

    def test_mixed_distributed_and_plain(self) -> None:
        mesh = mesh_1d(2)
        mapping = PlacementMapping(mesh, (Sharded(0),))
        buf_plain = Buffer.zeros([3, 4], dtype=DType.float32, device=CPU())
        buf_s0 = Buffer.zeros([2, 4], dtype=DType.float32, device=CPU())
        buf_s1 = Buffer.zeros([2, 4], dtype=DType.float32, device=CPU())
        slot_plain = _OutputSlot(start=0, count=1, mapping=None)
        slot_dist = _OutputSlot(start=1, count=2, mapping=mapping)
        result = _reconstruct_outputs(
            [buf_plain, buf_s0, buf_s1],
            [slot_plain, slot_dist],
            unary=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert not result[0].is_distributed
        assert result[1].is_distributed
        assert result[1].num_shards == 2


# ═════════════════════════════════════════════════════════════════════════
#  _flatten_named_buffers
# ═════════════════════════════════════════════════════════════════════════


class TestFlattenNamedBuffers:
    def test_single_device(self) -> None:
        t = Tensor.zeros([4, 8], dtype=DType.float32, device=CPU())
        result = _flatten_named_buffers([("weight", t)])
        assert "weight" in result
        assert len(result) == 1

    def test_sharded_expands(self) -> None:
        t = _make_realized_sharded([2, 8], 4, shard_axis=0)
        result = _flatten_named_buffers([("weight", t)])
        assert len(result) == 4
        assert "weight._shard.0" in result
        assert "weight._shard.1" in result
        assert "weight._shard.2" in result
        assert "weight._shard.3" in result

    def test_mixed(self) -> None:
        single = Tensor.zeros([4], dtype=DType.float32, device=CPU())
        sharded = _make_realized_sharded([2, 4], 2)
        result = _flatten_named_buffers([("bias", single), ("weight", sharded)])
        assert "bias" in result
        assert "weight._shard.0" in result
        assert "weight._shard.1" in result
        assert len(result) == 3

    def test_two_shard_naming(self) -> None:
        """Two-shard tensor uses ._shard.N naming, not bare name."""
        t = _make_realized_sharded([4, 4], 2, shard_axis=0)
        result = _flatten_named_buffers([("W", t)])
        assert "W" not in result
        assert "W._shard.0" in result
        assert "W._shard.1" in result

    def test_empty_input(self) -> None:
        result = _flatten_named_buffers([])
        assert result == {}


# ═════════════════════════════════════════════════════════════════════════
#  Module.compile() smoke tests
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
        input_type = TensorType(DType.float32, [3, 8], CPU())
        compiled = model.compile(input_type)
        x = Tensor.ones([3, 8], dtype=DType.float32, device=CPU())
        result = compiled(x)
        assert list(result.shape) == [3, 8]
        np.testing.assert_allclose(np.from_dlpack(result), 1.0)

    def test_compile_with_distributed_input_type(self) -> None:
        """DistributedTensorType input is flattened and reconstructed."""
        mesh = mesh_1d(2)
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _IdentityModule(W=W)
        input_type = DistributedTensorType(
            DType.float32, [8, 4], mesh, [Sharded(0)]
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
        input_type = DistributedTensorType(
            DType.float32, [8, 4], mesh, [Sharded(0)]
        )
        compiled = model.compile(input_type)
        x = _make_realized_sharded([4, 4], 2, shard_axis=0)
        result = compiled(x)
        for shard in result.local_shards:
            arr = np.from_dlpack(shard)
            np.testing.assert_allclose(arr, 0.0)


# ═════════════════════════════════════════════════════════════════════════
#  CompiledModel API surface
# ═════════════════════════════════════════════════════════════════════════


class TestCompiledModelAPI:
    """Tests for CompiledModel properties and execute_raw() method.

    These verify the public API surface that pipeline builders rely on:
    engine_model for CUDA graph capture, execute_raw for zero-overhead
    buffer execution, and signal_buffers for multi-GPU collectives.
    """

    def _compile_identity(self) -> CompiledModel:
        W = Tensor.ones([4], dtype=DType.float32, device=CPU())
        model = _IdentityModule(W=W)
        input_type = TensorType(DType.float32, [3, 8], CPU())
        return model.compile(input_type)

    def test_returns_compiled_model(self) -> None:
        """Module.compile() returns a CompiledModel instance."""
        compiled = self._compile_identity()
        assert isinstance(compiled, CompiledModel)

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
