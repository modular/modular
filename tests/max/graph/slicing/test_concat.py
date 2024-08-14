# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.concat tests."""

from random import Random

import pytest
from conftest import axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import DType, Graph, TensorType, ops
from max.graph.type import StaticDim


@given(
    base_type=st.shared(tensor_types(), key="type"),
    axis_sizes=...,
    axis=axes(st.shared(tensor_types(), key="type")),
)
def test_concat(base_type: TensorType, axis_sizes: list[StaticDim], axis: int):
    assume(len(axis_sizes) > 0)
    merged_size = sum(dim.dim for dim in axis_sizes)
    assume(merged_size < 2**63)

    input_types = []
    for i in range(len(axis_sizes)):
        shape = base_type.shape.copy()
        shape[axis] = axis_sizes[i]
        input_types.append(TensorType(base_type.dtype, shape))

    with Graph("concat", input_types=input_types) as graph:
        out = ops.concat(graph.inputs, axis)

        target_shape = base_type.shape.copy()
        target_shape[axis] = StaticDim(merged_size)
        assert out.shape == target_shape

        graph.output(out)
        graph._mlir_op.verify()


@given(
    base_type=st.shared(tensor_types(), key="type"),
    axis_sizes=...,
    axis=axes(st.shared(tensor_types(), key="type")),
    dtype=...,
    random=...,
)
def test_concat_bad_dtype(
    base_type: TensorType,
    axis_sizes: list[StaticDim],
    axis: int,
    dtype: DType,
    random: Random,
):
    assume(dtype != base_type.dtype)
    assume(len(axis_sizes) > 1)
    merged_size = sum(dim.dim for dim in axis_sizes)
    assume(merged_size < 2**63)

    input_types = []
    for i in range(len(axis_sizes)):
        shape = base_type.shape.copy()
        shape[axis] = axis_sizes[i]
        input_types.append(TensorType(base_type.dtype, shape))

    input_types[random.randint(0, len(input_types) - 1)].dtype = dtype

    with Graph("concat", input_types=input_types) as graph:
        with pytest.raises(ValueError, match="have a differing dtype"):
            out = ops.concat(graph.inputs, axis)
