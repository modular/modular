# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.repeat_interleave tests."""

import operator
from functools import reduce
from typing import Optional

import pytest
from conftest import axes, tensor_types
from hypothesis import assume, given, reject
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, StaticDim, TensorType, ops

shared_tensor_types = st.shared(tensor_types())


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1),
    axis=axes(shared_tensor_types),
)
def test_repeat_interleave(type: TensorType, repeats: int, axis: int):
    dim = type.shape[axis]
    assume(not isinstance(dim, StaticDim) or int(dim) * repeats < 2**63)
    with Graph("repeat_interleave", input_types=[type]) as graph:
        try:  # dims must fit into an int64
            _ = repeats * type.shape[axis]
        except ValueError:
            reject()  # test dim out of range
        out = ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis)
        target_shape = list(type.shape)
        target_shape[axis] *= repeats
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    axis=axes(shared_tensor_types),
)
def test_vector_repeats(type: TensorType, axis: int):
    dim = type.shape[axis]

    repeats_type = TensorType(DType.int64, [dim], device=DeviceRef.CPU())
    with Graph("repeat_interleave", input_types=[type, repeats_type]) as graph:
        out_dim = graph.current.unique_symbolic_dim("D")
        out = ops.repeat_interleave(
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            axis,
            out_dim=out_dim,
        )
        target_shape = list(type.shape)
        target_shape[axis] = out_dim
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1, max_value=2**63 - 1),
)
def test_repeat_interleave__no_axis(type: TensorType, repeats: int):
    static_product = reduce(operator.mul, type.shape.static_dims, repeats)
    assume(static_product < 2**63)
    with Graph("repeat_interleave", input_types=[type]) as graph:
        out = ops.repeat_interleave(graph.inputs[0].tensor, repeats)
        flat_size = reduce(operator.mul, type.shape, 1)
        target_shape = Shape([flat_size * repeats])
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    repeats=...,
    axis=st.one_of(axes(shared_tensor_types), st.none()),
)
def test_repeat_interleave__nonpositive_repeats(
    type: TensorType, repeats: int, axis: Optional[int]
):
    assume(repeats <= 0)
    with Graph("repeat_interleave", input_types=[type]) as graph:
        with pytest.raises(ValueError):
            ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis=axis)


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1),
    axis=...,
)
def test_repeat_interleave__axis_out_of_bounds(
    type: TensorType, repeats: int, axis: int
):
    assume(not -type.rank <= axis < type.rank)
    with Graph("repeat_interleave", input_types=[type]) as graph:
        with pytest.raises(ValueError):
            ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis=axis)
