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
"""Implements the fold operation."""

from algorithm import vectorize
from buffer.dimlist import DimList
from buffer.buffer import NDBuffer
from utils.index import Index, IndexList
from sys.info import simdwidthof


fn fold[
    dtype: DType,
    input_dim: DimList,
    output_dim: DimList,
    target: StaticString,
](
    input: NDBuffer[dtype, 3, MutableAnyOrigin, input_dim],
    output: NDBuffer[dtype, 4, MutableAnyOrigin, output_dim],
    output_size: IndexList[2],
    kernel_size: IndexList[2],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
) raises:
    """Folds array of sliding local blocks into a single output tensor.

    Args:
        input: Input tensor to fold, shape [N, C x kernel size, num_blocks].
        output: Output tensor to write to, shape [N, C, H, W].
        output_size: Spacial shape of the output tensor (H, W).
        kernel_size: Size of the sliding blocks.
        stride: Stride of the sliding blocks.
        dilation: Dilation of the sliding blocks.
        padding: 0-paddings to be added on both sides of the inputs.
    """

    constrained[target == "cpu", "Fold must be executed on CPU."]()

    if padding[0] < 0 or padding[1] < 0:
        raise Error("Padding must be non-negative.")

    if stride[0] <= 0 or stride[1] <= 0:
        raise Error("Stride must be positive.")

    if dilation[0] <= 0 or dilation[1] <= 0:
        raise Error("Dilation must be positive.")

    var N = output.dim[0]()
    var C = output.dim[1]()
    var H = output.dim[2]()
    var W = output.dim[3]()

    if output_size[0] != H or output_size[1] != W:
        raise Error("Output tensor size[2:] must be equal to output_size.")

    var channels_col = C * kernel_size[0] * kernel_size[1]

    if input.dim[1]() != channels_col:
        raise Error(
            "Input tensor channels must be equal to C * prod(kernel_size)."
        )
    var height_col = (
        (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
        // stride[0]
        + 1
    )

    var width_col = (
        (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        // stride[1]
        + 1
    )

    var num_blocks = input.dim[2]()

    var expected_blocks = height_col * width_col

    if num_blocks != expected_blocks:
        raise Error(
            "Input tensor must have the same number of blocks ("
            + String(num_blocks)
            + ") as the expected number of blocks ("
            + String(expected_blocks)
            + ")."
        )

    var kernel_w = kernel_size[1]
    var kernel_h = kernel_size[0]
    var padding_w = padding[1]
    var padding_h = padding[0]
    var dilation_w = dilation[1]
    var dilation_h = dilation[0]
    var stride_w = stride[1]
    var stride_h = stride[0]

    @parameter
    fn _fold_over_batch_cpu[simd_width: Int](batch: Int):
        var batch_input_offset = batch * channels_col * expected_blocks
        var batch_output_offset = batch * C * H * W
        var output_ptr = output.data + batch_output_offset
        var input_ptr = input.data + batch_input_offset
        for c in range(channels_col):
            var w_offset = c % kernel_w
            var h_offset = (c // kernel_w) % kernel_h
            var c_out = c // kernel_h // kernel_w

            for h in range(height_col):
                h_out = h * stride_h - padding_h + h_offset * dilation_h
                for w in range(width_col):
                    w_out = w * stride_w - padding_w + w_offset * dilation_w

                    if h_out >= 0 and h_out < H and w_out >= 0 and w_out < W:
                        var input_idx = (c * height_col + h) * width_col + w
                        var output_idx = (c_out * H + h_out) * W + w_out
                        before = output_ptr[output_idx]
                        output_ptr[output_idx] += input_ptr[input_idx]

    output.zero()
    vectorize[_fold_over_batch_cpu, simdwidthof[dtype]()](N)
