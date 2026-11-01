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
"""ops.resize tests."""

import re

import pytest
from conftest import GraphBuilder, dtypes, small_shapes
from hypothesis import assume, given
from hypothesis import strategies as st
from max._core.dialects import mo
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
def test_resize_valid(
    graph_builder: GraphBuilder, input_type: TensorType
) -> None:
    """Test valid resize operations."""
    with graph_builder(input_types=[input_type]) as graph:
        # Bicubic resizes height and width only, so double those.
        batch, channels, height, width = input_type.shape
        new_shape = Shape([batch, channels, height * 2, width * 2])

        out = ops.resize(
            graph.inputs[0].tensor,
            new_shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )

        assert out.dtype == input_type.dtype
        assert out.rank == input_type.rank
        assert out.shape == new_shape
        graph.output(out)


def test_resize_basic_upscale(graph_builder: GraphBuilder) -> None:
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


def test_resize_basic_downscale(graph_builder: GraphBuilder) -> None:
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
    graph_builder: GraphBuilder,
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


def test_resize_error_insufficient_rank(graph_builder: GraphBuilder) -> None:
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


@pytest.mark.parametrize("size", [[1, 1, 4, 4], [2, 3, 4, 4]])
def test_resize_bicubic_rejects_a_size_that_changes_planes(
    graph_builder: GraphBuilder, size: list[int]
) -> None:
    """The kernel resizes each of the input's N * C planes into the output's,
    so the size can't name other batch or channel counts."""
    input_type = TensorType(DType.float32, [1, 3, 8, 8], DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="batch and channel dims"):
            ops.resize_bicubic(graph.inputs[0].tensor, size)


def test_resize_bicubic_refuses_out_dims_with_a_static_size(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [1, 3, 8, 8], DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="out_dims"):
            ops.resize_bicubic(
                graph.inputs[0].tensor, [1, 3, 4, 4], out_dims=[1, 3, 2, 2]
            )


@pytest.mark.parametrize(
    "size_type",
    [
        # Torch's (H, W) convention.
        TensorType(DType.int64, [2], DeviceRef.CPU()),
        TensorType(DType.int64, ["k"], DeviceRef.CPU()),
        TensorType(DType.int32, [4], DeviceRef.CPU()),
        TensorType(DType.int64, [4], DeviceRef.GPU()),
    ],
)
def test_resize_bicubic_refuses_a_tensor_size_that_is_not_4_cpu_int64s(
    graph_builder: GraphBuilder, size_type: TensorType
) -> None:
    input_type = TensorType(DType.float32, [1, 3, 8, 8], DeviceRef.CPU())
    with graph_builder(input_types=[input_type, size_type]) as graph:
        x, size = (v.tensor for v in graph.inputs)
        with pytest.raises(ValueError, match="shape \\[4\\]"):
            ops.resize_bicubic(x, size, out_dims=[1, 3, "h", "w"])


def test_antialiased_resize_bicubic_takes_any_cpu(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [1, 3, 8, 8], DeviceRef.CPU(1))
    with graph_builder(input_types=[input_type]) as graph:
        ops.resize_bicubic(graph.inputs[0].tensor, [1, 3, 4, 4], antialias=True)


def test_antialiased_resize_bicubic_keeps_a_uint8_dtype(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.uint8, [1, 3, 8, 8], DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        out = ops.resize_bicubic(
            graph.inputs[0].tensor, [1, 3, 4, 16], antialias=True
        )
        assert out.type == TensorType(
            DType.uint8, [1, 3, 4, 16], DeviceRef.CPU()
        )


@pytest.mark.parametrize("dtype", [DType.int32, DType.float16])
def test_antialiased_resize_bicubic_refuses_other_dtypes(
    graph_builder: GraphBuilder, dtype: DType
) -> None:
    input_type = TensorType(dtype, [1, 3, 8, 8], DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="float32 or uint8"):
            ops.resize_bicubic(
                graph.inputs[0].tensor, [1, 3, 4, 4], antialias=True
            )


def test_resize_nearest(graph_builder: GraphBuilder) -> None:
    """Test resize with NEAREST interpolation delegates to resize_nearest."""
    input_type = TensorType(
        shape=[1, 3, 224, 224], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        out = ops.resize(
            graph.inputs[0].tensor,
            [1, 3, 448, 448],
            interpolation=ops.InterpolationMode.NEAREST,
        )

        assert out.dtype == input_type.dtype
        assert out.shape == Shape([1, 3, 448, 448])
        graph.output(out)


def test_resize_bilinear(graph_builder: GraphBuilder) -> None:
    """Test resize with BILINEAR interpolation delegates to resize_linear."""
    input_type = TensorType(
        shape=[1, 3, 224, 224], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        out = ops.resize(
            graph.inputs[0].tensor,
            [1, 3, 448, 448],
            interpolation=ops.InterpolationMode.BILINEAR,
        )

        assert out.dtype == input_type.dtype
        assert out.shape == Shape([1, 3, 448, 448])
        graph.output(out)


def test_coordinate_transform_mode_matches_mo() -> None:
    """Each mode keeps the code of its ``mo.CoordinateTransformMode`` case."""
    for mode in ops.CoordinateTransformMode:
        assert mo.CoordinateTransformMode(mode.value).name.upper() == mode.name


def test_resize_nearest_modes(graph_builder: GraphBuilder) -> None:
    """``resize_nearest`` takes the mode enums and rejects unknown codes."""
    input_type = TensorType(
        shape=[1, 3, 4, 4], dtype=DType.float32, device=DeviceRef.CPU()
    )

    with graph_builder(input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        out = ops.resize_nearest(
            x,
            [1, 3, 8, 8],
            coordinate_transform_mode=ops.CoordinateTransformMode.ASYMMETRIC,
            round_mode=ops.RoundMode.FLOOR,
        )
        assert out.shape == Shape([1, 3, 8, 8])
        with pytest.raises(ValueError):
            ops.resize_nearest(
                x,
                [1, 3, 8, 8],
                round_mode=4,  # type: ignore[arg-type]
            )
        graph.output(out)
