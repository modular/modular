# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from conftest import axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops

shared_types = st.shared(tensor_types(shapes=shapes(is_static=True)))
chunks = st.integers(min_value=1, max_value=4)


@given(input_type=shared_types, chunks=chunks, dim=axes(shared_types))
def test_chunk(input_type: TensorType, chunks: int, dim: int):
    assume(input_type.rank > 0)
    with Graph("chunk", input_types=[input_type]) as graph:
        target_shape = input_type.shape.static_dims
        n = target_shape[dim]
        assume(int(n) % chunks == 0)
        outs = ops.chunk(graph.inputs[0], chunks, dim=dim)
        chunk_size = (n + chunks - 1) // chunks
        for i in range(chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            expected_length = end - start
            assert int(outs[i].shape[dim]) == expected_length
        graph.output(outs[0])


@given(input_type=shared_types, chunks=chunks, dim=axes(shared_types))
def test_chunk_not_exact(input_type: TensorType, chunks: int, dim: int):
    assume(input_type.rank > 0)
    with Graph("chunk", input_types=[input_type]) as graph:
        target_shape = input_type.shape.static_dims
        n = target_shape[dim]
        assume(int(n) % chunks != 0)
        with pytest.raises(ValueError, match="must be exactly divisible"):
            outs = ops.chunk(graph.inputs[0], chunks, dim=dim)
