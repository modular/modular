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
"""ops.resize tests."""

import re

import pytest
from conftest import dtypes, small_shapes
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorType, ops

# Use constrained tensor types to avoid overflow in numpy conversion.
# And limit to rank 4 for NCHW format.
constrained_tensor_types = st.builds(
    TensorType,
    dtypes,
    small_shapes(min_rank=4, max_rank=4),  # NCHW format only
    st.just(DeviceRef.CPU()),
)
input_types = st.shared(constrained_tensor_types)


@given(input_type=input_types)
def test_resize_valid(graph_builder, input_type: TensorType) -> None:  # noqa: ANN001
    """Test valid resize operations."""
    with graph_builder(input_types=[input_type]) as graph:
        # Create a valid size shape - just double all dimensions.
        new_shape = Shape(dim * 2 for dim in input_type.shape)

        out = ops.resize(
            graph.inputs[0].tensor,
            new_shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )

        assert out.dtype == input_type.dtype
        assert out.rank == input_type.rank
        assert out.shape == new_shape
        graph.output(out)


def test_resize_basic_upscale(graph_builder) -> None:  # noqa: ANN001
    """Test basic resize upscaling."""
    input_type = TensorType(
        shape=[1, 3, 224, 224], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        # Upscale to 448x448
        out = ops.resize(
            graph.inputs[0].tensor,
            [1, 3, 448, 448],
            interpolation=ops.InterpolationMode.BICUBIC,
        )

        assert out.dtype == input_type.dtype
        assert out.shape == Shape([1, 3, 448, 448])
        graph.output(out)


def test_resize_basic_downscale(graph_builder) -> None:  # noqa: ANN001
    """Test basic resize downscaling."""
    input_type = TensorType(
        shape=[2, 3, 256, 256], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        # Downscale to 128x128
        out = ops.resize(
            graph.inputs[0].tensor,
            [2, 3, 128, 128],
            interpolation=ops.InterpolationMode.BICUBIC,
        )

        assert out.dtype == input_type.dtype
        assert out.shape == Shape([2, 3, 128, 128])
        graph.output(out)


@given(input_type=input_types, resize_shape=...)
def test_resize_error_size_wrong_length(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    resize_shape: Shape,
) -> None:
    """Test error when size has wrong number of elements."""
    assume(input_type.rank != resize_shape.rank)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "shape must have 4 elements for NCHW format (batch, channels, height, width)"
            ),
        ):
            ops.resize(
                graph.inputs[0].tensor,
                resize_shape,
                interpolation=ops.InterpolationMode.BICUBIC,
            )


def test_resize_error_insufficient_rank(graph_builder) -> None:  # noqa: ANN001
    """Test error when input has insufficient rank."""
    # Create a rank-2 tensor
    input_type = TensorType(
        shape=[224, 224], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Input tensor must have rank 4 (NCHW format) for resize operation, but got rank 2"
            ),
        ):
            ops.resize(
                graph.inputs[0].tensor,
                [448, 448],
                interpolation=ops.InterpolationMode.BICUBIC,
            )


def test_resize_error_unsupported_interpolation(graph_builder) -> None:  # noqa: ANN001
    """Test error when using unsupported interpolation mode."""
    input_type = TensorType(
        shape=[1, 3, 224, 224], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(
            NotImplementedError,
            match=re.escape("Interpolation mode bilinear is not yet supported"),
        ):
            ops.resize(
                graph.inputs[0].tensor,
                [1, 3, 448, 448],
                interpolation=ops.InterpolationMode.BILINEAR,
            )
