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
"""ops.conv_transpose tests."""

from math import sqrt

import numpy as np
import pytest
from conftest import MAX_INT32, static_dims, tensor_types
from hypothesis import assume, given, reject
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, Weight, ops

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
output_padding_type = st.tuples(sized_int, sized_int)


@given(
    x_type=static_tensor_type,
    filter_type=static_tensor_type,
    stride=stride_type,
    padding=padding_type,
    output_padding=output_padding_type,
)
def test_conv_transpose_valid(
    x_type: TensorType,
    filter_type: TensorType,
    stride,  # noqa: ANN001
    padding,  # noqa: ANN001
    output_padding,  # noqa: ANN001
) -> None:
    assume(filter_type.shape[0] <= x_type.shape[1])
    assume(filter_type.shape[1] <= x_type.shape[2])

    with Graph("conv2d_transpose", input_types=[x_type, filter_type]) as graph:
        try:
            padding = (0, 0, 0, 0)
            out = ops.conv2d_transpose(
                x=graph.inputs[0].tensor,
                filter=graph.inputs[1].tensor,
                stride=stride,
                dilation=(1, 1),
                padding=padding,
                output_paddings=output_padding,
            )
        except ValueError:
            reject()

        output_height = (
            (x_type.shape[1] - 1) * stride[0]
            - padding[0]
            - padding[1]
            + filter_type.shape[0]
            - 1
            + output_padding[0]
            + 1
        )

        output_width = (
            (x_type.shape[2] - 1) * stride[1]
            - padding[2]
            - padding[3]
            + filter_type.shape[1]
            - 1
            + output_padding[1]
            + 1
        )

        assert out.shape == [
            x_type.shape[0],
            output_height,
            output_width,
            filter_type.shape[2],
        ]
        graph.output(out)


def test_conv_dtype_promote_np() -> None:
    x_type = TensorType(
        DType.bfloat16, [1, 128, 128, 4], device=DeviceRef.CPU()
    )
    filter_shape = [3, 3, 5, 4]
    filter = np.ones(filter_shape, dtype=np.float32)
    with Graph("conv_transpose", input_types=[x_type]) as graph:
        out = ops.conv2d_transpose(
            graph.inputs[0].tensor,
            filter,
        )
        # The numpy filter has a weak dtype. This all resolves happily.
        assert out.dtype == DType.bfloat16
        graph.output(out)


def test_conv_dtype_promote_weight() -> None:
    x_type = TensorType(
        DType.bfloat16, [1, 128, 128, 4], device=DeviceRef.CPU()
    )
    filter_shape = [3, 3, 5, 4]
    filter = Weight(
        "filter",
        dtype=DType.bfloat16,
        shape=filter_shape,
        device=DeviceRef.CPU(),
    )
    with Graph("conv", input_types=[x_type]) as graph:
        out = ops.conv2d_transpose(
            graph.inputs[0],
            filter,
        )
        # Both input and filter dtype exactly match.
        assert out.dtype == DType.bfloat16
        graph.output(out)


def test_conv_dtype_promote_weight_success() -> None:
    x_type = TensorType(
        DType.bfloat16, [1, 128, 128, 4], device=DeviceRef.CPU()
    )
    filter_shape = [3, 3, 5, 4]
    filter = Weight(
        "filter",
        dtype=DType.float32,
        shape=filter_shape,
        device=DeviceRef.CPU(),
    )
    with Graph("conv", input_types=[x_type]) as graph:
        # Both the input and weight have strong dtypes. Conv requires them to match.
        out = ops.conv2d_transpose(
            graph.inputs[0].tensor,
            filter,
        )
        assert out.dtype == DType.float32


def test_conv_dtype_promote_weight_failed() -> None:
    x_type = TensorType(DType.int32, [1, 128, 128, 4], device=DeviceRef.CPU())
    filter_shape = [3, 3, 5, 4]
    filter = Weight(
        "filter",
        dtype=DType.float16,
        shape=filter_shape,
        device=DeviceRef.CPU(),
    )
    with Graph("conv", input_types=[x_type]) as graph:
        # Both the input and weight have strong dtypes. Conv requires them to match.
        with pytest.raises(
            ValueError,
            match=(
                "Failed to resolve valid dtype: Unsafe cast from si32 to f32."
                " Insert an explicit cast op if this conversion is wanted"
            ),
        ):
            out = ops.conv2d(
                graph.inputs[0].tensor,
                filter,
            )


def test_conv_symbolic_shapes() -> None:
    input_type = TensorType(
        DType.bfloat16,
        [1, "height", "width", "channels"],
        device=DeviceRef.CPU(),
    )
    filter_type = TensorType(
        DType.bfloat16, [16, 16, 3, 1024], device=DeviceRef.CPU()
    )

    strides = (16, 16)
    dilations = (1, 1)
    paddings = (0, 0, 0, 0)
    output_paddings = (0, 0)

    with Graph("symbolic_conv", input_types=[input_type, filter_type]) as graph:
        out = ops.conv2d_transpose(
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            strides,
            dilations,
            paddings,
            output_paddings,
        )

        graph.output(out)
