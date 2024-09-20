# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""mutable ops tests."""

import pytest
from conftest import buffer_types, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    Graph,
    TensorType,
    TensorValue,
    ops,
)


@given(buffer_type=...)
def test_buffer_value(buffer_type: BufferType):
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


shared_dtypes = st.shared(st.from_type(DType))

shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_type = tensor_types(shapes=shared_shapes, dtypes=shared_dtypes)
buffer_type = buffer_types(shapes=shared_shapes, dtypes=shared_dtypes)


@given(buffer_type=...)
def test_load_buffer(buffer_type: BufferType):
    with Graph(
        "buffer_load",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0]
        chain_0 = graph._current_chain
        y = ops.load_buffer(buffer)
        chain_1 = graph._current_chain

        assert y.shape == buffer.shape
        assert y.dtype == buffer.dtype
        assert isinstance(y, TensorValue)

        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "mo.chain.create" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_store_buffer(tensor_type: TensorType, buffer_type: BufferType):
    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0]
        buffer = graph.inputs[1]
        chain_0 = graph._current_chain
        ops.store_in_buffer(tensor, buffer)
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
def test_load_store_buffer(buffer_type: BufferType):
    with Graph(
        "buffer_load",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0]
        chain_0 = graph._current_chain
        tensor = ops.load_buffer(buffer)
        chain_1 = graph._current_chain

        assert tensor.shape == buffer.shape
        assert tensor.dtype == buffer.dtype
        assert isinstance(tensor, TensorValue)

        # Check the chain is updated.
        assert chain_0 != chain_1

        ops.store_in_buffer(tensor, buffer)
        chain_2 = graph._current_chain

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype

        # Check the chain is updated.
        assert chain_0 != chain_2
        assert chain_1 != chain_2

        graph.output()
        graph._mlir_op.verify()
        assert "mo.chain.create" in str(graph)
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)


# TODO(MSDK-960): test that the chain is working correctly.
# TODO(MSDK-960): test load -> element-wise ops -> store.
