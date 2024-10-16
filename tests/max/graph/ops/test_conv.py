# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.conv2d tests."""
from math import sqrt

from conftest import (
    MAX_INT32,
    static_dims,
    static_known_shape_size,
    tensor_types,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))
static_tensor_type = tensor_types(
    dtypes=shared_dtypes,
    shapes=(
        st.lists(
            static_dims(min=1, max=int(sqrt(sqrt(MAX_INT32)))),
            min_size=4,
            max_size=4,
        )
    ),
)

sized_int = st.integers(min_value=0, max_value=10)
pos_int = st.integers(min_value=1, max_value=10)

padding_type = st.tuples(sized_int, sized_int, sized_int, sized_int)
stride_type = st.tuples(pos_int, pos_int)


@given(
    x_type=static_tensor_type,
    filter_type=static_tensor_type,
    stride=stride_type,
    padding=padding_type,
)
def test_conv_valid(
    x_type: TensorType, filter_type: TensorType, stride, padding
):
    # TODO(GRA-1015): remove the next two assumptions.
    assume(static_known_shape_size(filter_type.shape) <= MAX_INT32)
    assume(static_known_shape_size(x_type.shape) <= MAX_INT32)
    assume(filter_type.shape[0] <= x_type.shape[1])
    assume(filter_type.shape[1] <= x_type.shape[2])
    with Graph("conv", input_types=[x_type, filter_type]) as graph:
        out = ops.conv2d(
            graph.inputs[0],
            graph.inputs[1],
            stride=stride,
            padding=padding,
        )
        output_height = (
            x_type.shape[1]
            - filter_type.shape[0]
            + padding[0]
            + padding[1]
            + stride[0]
        ) // stride[0]
        output_width = (
            x_type.shape[2]
            - filter_type.shape[1]
            + padding[2]
            + padding[3]
            + stride[1]
        ) // stride[1]
        assert out.shape == [
            x_type.shape[0],
            output_height,
            output_width,
            filter_type.shape[3],
        ]
        graph.output(out)
