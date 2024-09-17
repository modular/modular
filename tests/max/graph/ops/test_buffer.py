# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from conftest import buffer_types, tensor_types
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
tensor_type = tensor_types(dtypes=shared_dtypes)
buffer_type = buffer_types(dtypes=shared_dtypes)


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

        # check the chain is updated.
        assert chain_0 == chain_0
        assert chain_1 == chain_1
        assert chain_0 != chain_1
        graph.output()


# TODO(MSDK-960): test that the chain is working correctly.
# TODO(MSDK-960): test load -> element-wise ops -> store.
