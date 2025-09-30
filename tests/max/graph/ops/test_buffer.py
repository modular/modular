# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""mutable ops tests."""

import pytest
from conftest import buffer_types, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max import mlir
from max._core.dialects import mo
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    Graph,
    TensorType,
    TensorValue,
    Value,
    _ChainType,
    _ChainValue,
    ops,
)

shared_dtypes = st.shared(st.from_type(DType))
shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_type = tensor_types(shapes=shared_shapes, dtypes=shared_dtypes)
buffer_type = buffer_types(shapes=shared_shapes, dtypes=shared_dtypes)


@given(buffer_type=...)
def test_mlir_type_checking(buffer_type: BufferType) -> None:
    with Graph(
        "buffer",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0]
        type = buffer.type
        assert isinstance(buffer, BufferValue)
        assert type == buffer_type
        assert not isinstance(buffer, mlir.Value)
        assert isinstance(buffer._mlir_value.type, mo.BufferType)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_value_constructor(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    with Graph(
        "buffer_store",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        buffer = Value.from_mlir(graph.inputs[1]._mlir_value)
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = Value.from_mlir(graph.inputs[0]._mlir_value)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)
        with pytest.raises(Exception):
            TensorValue.from_mlir(graph.inputs[1]._mlir_value)
        with pytest.raises(Exception):
            BufferValue.from_mlir(graph.inputs[0]._mlir_value)

        buffer = BufferValue(graph.inputs[1]._mlir_value)
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = TensorValue(graph.inputs[0]._mlir_value)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)

        with pytest.raises(AssertionError):
            BufferValue(graph.inputs[0]._mlir_value)
        with pytest.raises(AssertionError):
            TensorValue(graph.inputs[1]._mlir_value)

        buffer = BufferValue(graph.inputs[1])
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = TensorValue(graph.inputs[0].tensor)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)

        with pytest.raises(TypeError):
            BufferValue(graph.inputs[0])
        with pytest.raises(TypeError):
            TensorValue(graph.inputs[1].tensor)

        with pytest.raises(TypeError):
            TensorValue(0)

        with pytest.raises(TypeError):
            BufferValue(0)  # type: ignore


# buffer and tensor inputs share dtype and shape
@given(buffer_type=...)
def test_load(buffer_type: BufferType) -> None:
    with Graph(
        "buffer_load",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0].buffer
        chain_0 = graph._current_chain
        assert isinstance(chain_0, _ChainValue)
        assert isinstance(chain_0.type, _ChainType)

        y = ops.buffer_load(buffer)
        chain_1 = graph._current_chain

        assert isinstance(chain_1, _ChainValue)
        assert isinstance(chain_1.type, _ChainType)

        assert y.shape == buffer.shape
        assert y.dtype == buffer.dtype
        assert isinstance(y, TensorValue)
        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "mo.chain.create" in str(graph)


# buffer and tensor inputs share dtype and shape
@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_store(tensor_type: TensorType, buffer_type: BufferType) -> None:
    with Graph(
        "buffer_store",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph._current_chain
        ops.buffer_store(buffer, tensor)
        chain_1 = graph._current_chain

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype

        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.store" in str(graph)
        assert "mo.chain.create" in str(graph)


@given(buffer_type=...)
def test_load_store(buffer_type: BufferType) -> None:
    with Graph(
        "buffer_load_store",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0].buffer
        chain_0 = graph._current_chain
        tensor = ops.buffer_load(buffer)
        chain_1 = graph._current_chain

        assert tensor.shape == buffer.shape
        assert tensor.dtype == buffer.dtype
        assert isinstance(tensor, TensorValue)
        assert chain_0 != chain_1

        ops.buffer_store(buffer, tensor)
        chain_2 = graph._current_chain

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        assert chain_0 != chain_2
        assert chain_1 != chain_2

        graph.output()
        graph._mlir_op.verify()
        assert "mo.chain.create" in str(graph)
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_load_store_ellipsis_slice(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph._current_chain
        buffer[...] = tensor + buffer[...]
        chain_1 = graph._current_chain

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)
        assert "rmo.mo.mutable.store.slice" not in str(graph)
        assert "mo.chain.create" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_load_store_slice(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph._current_chain
        buffer[0] = tensor[0] + buffer[0]
        chain_1 = graph._current_chain

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)
        assert "rmo.mo.mutable.store.slice" in str(graph)
        assert "mo.chain.create" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_no_implicit_load(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0]
        buffer = graph.inputs[1]

        with pytest.raises(TypeError):  # binary ops
            y = tensor + buffer  # type: ignore

        with pytest.raises(TypeError):  # unary ops
            y = abs(buffer)  # type: ignore

        assert "rmo.mo.mutable.load" not in str(graph)
        assert "rmo.mo.slice" not in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_prints_with_buffer_ops(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    with Graph(
        "debug_prints_and_mutable_ops",
        input_types=[buffer_type, tensor_type],
    ) as graph:
        buffer: BufferValue = graph.inputs[0].buffer
        tensor: TensorValue = graph.inputs[1].tensor

        chain_0 = graph._current_chain

        tensor.print()
        chain_1 = graph._current_chain

        x = buffer[...]
        chain_2 = graph._current_chain

        x.print()
        chain_3 = graph._current_chain

        ops.buffer_store(buffer, tensor)
        chain_3 = graph._current_chain

        graph.output()

        assert chain_0 != chain_1
        assert chain_1 != chain_2
        assert chain_2 != chain_3


# TODO(MSDK-960): test that the chain is working correctly.
# TODO(MSDK-960): test load -> element-wise ops -> store.
