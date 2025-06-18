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


from algorithm import elementwise
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList


fn fold[
    dtype: DType,
    input_dim: DimList,
    output_dim: DimList, //,
    stride: Tuple[Int, Int],
    dilation: Tuple[Int, Int],
    padding: Tuple[Int, Int],
    target: StaticString,
](
    input: NDBuffer[dtype, 3, MutableAnyOrigin, input_dim],
    output: NDBuffer[dtype, 4, MutableAnyOrigin, output_dim],
    output_size: IndexList[2],
    kernel_size: IndexList[2],
    ctx: DeviceContextPtr,
) raises:
    """Folds array of sliding local blocks into a single output tensor.

    Parameters:
        dtype: The data type for the input and output.
        input_dim: The static shape of the input NDBuffer.
        output_dim: The static shape of the output NDBuffer.
        stride: Stride of the sliding blocks.
        dilation: Dilation of the sliding blocks.
        padding: 0-paddings to be added on both sides of the inputs.
        target: The target architecture to compile for.

    Args:
        input: Input tensor to fold, shape [N, C x kernel size, num_blocks].
        output: Output tensor to write to, shape [N, C, H, W].
        output_size: Spatial shape of the output tensor (H, W).
        kernel_size: Size of the sliding blocks.
        ctx: DeviceContextPtr.
    """

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
        H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1

    var width_col = (
        W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

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
    alias dilation_w = dilation[1]
    alias dilation_h = dilation[0]
    alias stride_w = stride[1]
    alias stride_h = stride[0]

    @always_inline
    @parameter
    @__copy_capture(
        kernel_w,
        kernel_h,
        height_col,
        width_col,
    )
    fn fold_fn[width: Int, rank_: Int](idx_arg: IndexList[rank_]):
        constrained[rank_ == 4, "fold_fn: rank must be 4"]()
        var idx = rebind[IndexList[4]](idx_arg)

        var batch = idx[0]
        var channel = idx[1]
        var h_out = idx[2]
        var w_out = idx[3]

        var output_val = Scalar[dtype](0)

        # The span of the kernel in the output tensor.
        var kernel_span_w = (kernel_w - 1) * dilation_w + 1
        var kernel_span_h = (kernel_h - 1) * dilation_h + 1

        # Given the position in the output tensor (h_out, w_out), compute the
        # start and end of the kernel patches that might overlap with this position.
        var h_start = max(0, (h_out - kernel_span_h) // stride_h + 1)
        var w_start = max(0, (w_out - kernel_span_w) // stride_w + 1)
        var h_end = min(h_out // stride_h + 1, height_col)
        var w_end = min(w_out // stride_w + 1, width_col)

        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                # compute the relative position of current position in the
                # kernel patch.
                var h_offset = h_out - h * stride_h
                var w_offset = w_out - w * stride_w

                # Check if the current position is covered by the patch.
                if h_offset % dilation_h == 0 and w_offset % dilation_w == 0:
                    h_offset = h_offset // dilation_h
                    w_offset = w_offset // dilation_w

                    var channel_offset = channel * kernel_h * kernel_w
                    var kernel_offset = h_offset * kernel_w + w_offset
                    var patch_offset = h * width_col + w

                    # Load and accumulate
                    output_val += input[
                        batch, channel_offset + kernel_offset, patch_offset
                    ]

        output.store(idx, output_val)

    var dispatch_shape = IndexList[4](N, C, H, W)
    elementwise[
        func=fold_fn,
        simd_width=1,
        target=target,
        _trace_description="fold_fn",
    ](dispatch_shape, ctx)


fn fold_shape[
    dtype: DType, input_dim: DimList
](
    input: NDBuffer[dtype, 3, MutableAnyOrigin, input_dim],
    output_size: IndexList[2],
    kernel_size: IndexList[2],
) raises -> IndexList[4]:
    """Returns the shape of the output tensor of the fold operation."""
    var output_shape = IndexList[4]()
    output_shape[0] = input.dim[0]()
    output_shape[1] = input.dim[1]() // (kernel_size[0] * kernel_size[1])
    output_shape[2] = output_size[0]
    output_shape[3] = output_size[1]
    return output_shape
