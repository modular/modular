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
"""Op implementation for conv3d."""

from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

# TODO: Fails if we use 2**63 - 1 as max value
positive_dims = st.integers(min_value=1, max_value=2**32)
strides = st.tuples(*[positive_dims] * 3)
paddings = st.tuples(*[positive_dims] * 6)


@given(
    batch_size=positive_dims,
    d=positive_dims,
    h=positive_dims,
    w=positive_dims,
    in_c=positive_dims,
    out_c=positive_dims,
    k_d=positive_dims,
    k_h=positive_dims,
    k_w=positive_dims,
    stride=strides,
    padding=paddings,
)
def test_conv3d(
    batch_size: int,
    d: int,
    h: int,
    w: int,
    in_c: int,
    out_c: int,
    k_d: int,
    k_h: int,
    k_w: int,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
) -> None:
    """Test 3D convolution with various parameters to verify shape correctness."""
    dilation = (1, 1, 1)  # We don't support dilation yet

    # Calculate expected output dimensions
    out_d = (
        d + padding[0] + padding[1] - dilation[0] * (k_d - 1) - 1
    ) // stride[0] + 1
    out_h = (
        h + padding[2] + padding[3] - dilation[1] * (k_h - 1) - 1
    ) // stride[1] + 1
    out_w = (
        w + padding[4] + padding[5] - dilation[2] * (k_w - 1) - 1
    ) // stride[2] + 1

    # Ensure valid output dimensions
    assume(out_d > 0 and out_h > 0 and out_w > 0)

    # Create input and filter tensor types
    input_type = TensorType(
        dtype=DType.float32,
        shape=[batch_size, d, h, w, in_c],
        device=DeviceRef.CPU(),
    )
    filter_type = TensorType(
        dtype=DType.float32,
        shape=[k_d, k_h, k_w, in_c, out_c],
        device=DeviceRef.CPU(),
    )

    # Set up graph with input and filter tensors
    with Graph(
        "conv3d_test",
        input_types=[input_type, filter_type],
    ) as graph:
        x_tensor = graph.inputs[0].tensor
        filter_tensor = graph.inputs[1].tensor

        result = ops.conv3d(
            x=x_tensor,
            filter=filter_tensor,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=1,
            bias=None,
        )

        expected_shape = [
            batch_size,
            out_d,
            out_h,
            out_w,
            out_c,
        ]
        assert list(result.shape) == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}"
        )
