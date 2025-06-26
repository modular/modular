# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.fold tests."""

import math

import pytest
from conftest import dtypes
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, TensorType, ops

valid_dim = st.integers(min_value=1, max_value=1024)


@given(
    dtype=dtypes,
    batch=valid_dim,
    channel=valid_dim,
    output_size=st.tuples(valid_dim, valid_dim),
    kernel_size=st.tuples(valid_dim, valid_dim),
    stride=st.tuples(valid_dim, valid_dim),
    dilation=st.tuples(valid_dim, valid_dim),
    padding=st.tuples(valid_dim, valid_dim),
)
def test_fold(
    graph_builder,
    dtype: DType,
    batch: int,
    channel: int,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int],
):
    """Padding by nothing does not change the type."""
    # Create valid input shape from input values.
    L = 1
    for n, (o, k) in enumerate(zip(output_size, kernel_size)):
        L_d = int(
            (o + 2 * padding[n] - dilation[n] * (k - 1) - 1) // stride[n] + 1
        )
        L *= L_d
    assume(L > 0)

    dim = channel * math.prod(kernel_size)
    input_dim = (batch, dim, L)
    input_type = TensorType(dtype, input_dim, DeviceRef.CPU())

    # Build graph and check output shape and dtype.
    with graph_builder(input_types=[input_type]) as graph:
        out = ops.fold(
            graph.inputs[0].tensor,
            output_size,
            kernel_size,
            stride,
            dilation,
            padding,
        )
        assert out.type.shape == [
            batch,
            channel,
            output_size[0],
            output_size[1],
        ]
        assert out.type.dtype == input_type.dtype


@given(
    dtype=dtypes,
    batch=valid_dim,
    channel=valid_dim,
    output_size=st.tuples(valid_dim, valid_dim),
    kernel_size=st.tuples(valid_dim, valid_dim),
    stride=st.tuples(valid_dim, valid_dim),
    dilation=st.tuples(valid_dim, valid_dim),
    padding=st.tuples(valid_dim, valid_dim),
)
def test_fold_invalid_inputs(
    graph_builder,
    dtype: DType,
    batch: int,
    channel: int,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int],
):
    """Padding by nothing does not change the type."""
    # Create valid input shape from input values.
    L = 1
    for n, (o, k) in enumerate(zip(output_size, kernel_size)):
        L_d = int(
            (o + 2 * padding[n] - dilation[n] * (k - 1) - 1) // stride[n] + 1
        )
        L *= L_d
    assume(L > 0)
    channel_dim = channel * math.prod(kernel_size)

    # Invalid L dimension.
    invalid_input_dim = (batch, channel_dim, L + 1)
    input_type = TensorType(dtype, invalid_input_dim, DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(
            ValueError, match=".*must match the calculated number of blocks.*"
        ):
            _ = ops.fold(
                graph.inputs[0].tensor,
                output_size,
                kernel_size,
                stride,
                dilation,
                padding,
            )

    # Invalid channel dimension.
    assume(math.prod(kernel_size) > 1)
    invalid_input_dim = (batch, channel_dim + 1, L)
    input_type = TensorType(dtype, invalid_input_dim, DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(
            ValueError,
            match=".*must be a multiple of the product of the total kernel size.*",
        ):
            _ = ops.fold(
                graph.inputs[0].tensor,
                output_size,
                kernel_size,
                stride,
                dilation,
                padding,
            )
