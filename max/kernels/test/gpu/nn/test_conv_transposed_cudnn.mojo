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
from buffer import DimList
from gpu.host import DeviceContext
from gpu.host.info import Vendor
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from layout import Layout, LayoutTensor, RuntimeLayout
from nn.conv_transpose import conv_transpose_naive, conv_transposed_cudnn
from testing import assert_almost_equal

from utils.index import Index, IndexList

alias dtype = DType.float32


fn test_conv_transposed_cudnn[
    input_len: Int,
    kernel_len: Int,
    in_channels: Int = 1,
    out_channels: Int = 1,
    stride_val: Int = 1,
    dilation_val: Int = 1,
    pad_val: Int = 0,
    dtype: DType = DType.float32,
](ctx: DeviceContext,) raises:
    """
    Fixed 1-D transposed-convolution test with correct QRSFC kernel layout.
    """

    print(
        "input_len=",
        input_len,
        ", kernel_size=",
        kernel_len,
        ", in_channels=",
        in_channels,
        ", out_channels=",
        out_channels,
        ", stride=",
        stride_val,
        ", pad=",
        pad_val,
        ", dilation=",
        dilation_val,
        ")",
    )

    alias layout_unknown_5d = Layout.row_major[5]()

    # Shapes.
    alias input_shape4d = DimList(1, 1, input_len, in_channels)
    alias filter_shape4d = DimList(1, kernel_len, out_channels, in_channels)
    alias output_len = (
        stride_val * (input_len - 1)
        + dilation_val * (kernel_len - 1)
        - 2 * pad_val
        + 1
    )
    alias output_shape4d = DimList(1, 1, output_len, out_channels)
    alias input_layout5d = Layout.row_major(1, 1, 1, input_len, in_channels)
    alias filter_layout5d = Layout.row_major(
        1, 1, kernel_len, out_channels, in_channels
    )
    alias output_layout5d = Layout.row_major(1, 1, 1, output_len, out_channels)

    # Create host buffers using HostNDBuffer
    var input_host = HostNDBuffer[dtype, 4, input_shape4d](input_shape4d)
    var filter_host = HostNDBuffer[dtype, 4, filter_shape4d](filter_shape4d)
    var output_host = HostNDBuffer[dtype, 4, output_shape4d](output_shape4d)
    var output_ref_host = HostNDBuffer[dtype, 4, output_shape4d](output_shape4d)

    random(input_host.tensor)
    random(filter_host.tensor)

    # Parameters (1-D ⇒ only W dimension varies).
    var stride = Index(1, 1, stride_val)
    var dilation = Index(1, 1, dilation_val)
    var pad_d = Index(0, 0)  # depth – none in 1-D
    var pad_h = Index(0, 0)  # height – none in 1-D
    var pad_w = Index(pad_val, pad_val)  # width padding (symmetric)

    # Execute naive reference implementation.
    conv_transpose_naive[dtype](
        LayoutTensor[dtype, layout_unknown_5d](
            output_ref_host.tensor.data,
            RuntimeLayout[layout_unknown_5d].row_major(
                IndexList[5](1, 1, 1, output_len, out_channels)
            ),
        ),
        LayoutTensor[dtype, layout_unknown_5d](
            input_host.tensor.data,
            RuntimeLayout[layout_unknown_5d].row_major(
                IndexList[5](1, 1, 1, input_len, in_channels)
            ),
        ),
        LayoutTensor[dtype, layout_unknown_5d](
            filter_host.tensor.data,
            RuntimeLayout[layout_unknown_5d].row_major(
                IndexList[5](1, 1, kernel_len, out_channels, in_channels)
            ),
        ),
        stride,
        dilation,
        pad_d,  # D
        pad_h,  # H
        pad_w,  # W
    )

    # -------------------------------------------------------------
    # 2. Run the same transposed-convolution via cuDNN backward-data
    # -------------------------------------------------------------

    # Convert input/output data from NHWC to NCHW layout for cuDNN
    # Convert filter data from QRSFC to CFHW layout for cuDNN
    alias input_shape4d_nchw = DimList(1, in_channels, 1, input_len)
    alias output_shape4d_nchw = DimList(1, out_channels, 1, output_len)
    alias filter_shape4d_nchw = DimList(
        in_channels, out_channels, 1, kernel_len
    )

    var input_nchw_host = HostNDBuffer[dtype, 4, input_shape4d_nchw](
        input_shape4d_nchw
    )
    for w in range(input_len):
        for c in range(in_channels):
            input_nchw_host.tensor[0, c, 0, w] = input_host.tensor[0, 0, w, c]

    var filter_nchw_host = HostNDBuffer[dtype, 4, filter_shape4d_nchw](
        filter_shape4d_nchw
    )
    for r in range(1):
        for s in range(kernel_len):
            for f in range(out_channels):
                for c in range(in_channels):
                    filter_nchw_host.tensor[c, f, r, s] = filter_host.tensor[
                        r, s, f, c
                    ]

    var output_nchw_host = HostNDBuffer[dtype, 4, output_shape4d_nchw](
        output_shape4d_nchw
    )

    # Create device buffers using enqueue operations
    var d_input = DeviceNDBuffer[dtype, 4, input_shape4d_nchw](
        input_shape4d_nchw, ctx=ctx
    )
    var d_filter = DeviceNDBuffer[dtype, 4, filter_shape4d_nchw](
        filter_shape4d_nchw, ctx=ctx
    )
    var d_output = DeviceNDBuffer[dtype, 4, output_shape4d_nchw](
        output_shape4d_nchw, ctx=ctx
    )
    ctx.enqueue_copy(d_input.buffer, input_nchw_host.tensor.data)
    ctx.enqueue_copy(d_filter.buffer, filter_nchw_host.tensor.data)

    var stride_hw = Index(1, stride_val)
    var dilation_hw = Index(1, dilation_val)
    var padding_hw = Index(0, pad_val)

    alias layout_unknown_4d = Layout.row_major[4]()

    # Invoke cuDNN helper.
    conv_transposed_cudnn[dtype, dtype, dtype](
        LayoutTensor[dtype, layout_unknown_4d](
            d_input.buffer,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](1, in_channels, 1, input_len)
            ),
        ),  # dy (input grad)
        LayoutTensor[dtype, layout_unknown_4d](
            d_filter.buffer,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](in_channels, out_channels, 1, kernel_len)
            ),
        ),  # w (filter)
        LayoutTensor[dtype, layout_unknown_4d](
            d_output.buffer,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](1, out_channels, 1, output_len)
            ),
        ),  # dx (output)
        stride_hw,
        dilation_hw,
        padding_hw,
        ctx,
    )

    # Copy result back to host using enqueue_copy
    ctx.enqueue_copy(output_nchw_host.tensor.data, d_output.buffer)

    # -------------------------------------------------------------
    # 3. Compare naive vs cuDNN results
    # -------------------------------------------------------------

    # verifying results
    output_ref_host_buf = output_ref_host.tensor
    output_nchw_host_buf = output_nchw_host.tensor
    for w in range(output_len):
        for f in range(out_channels):
            assert_almost_equal(
                output_ref_host_buf[0, 0, w, f],
                output_nchw_host_buf[0, f, 0, w],
                rtol=0.0001,
            )
    print("Succeed")

    # Clean up - device buffers will be cleaned up automatically
    _ = input_host
    _ = filter_host
    _ = output_host
    _ = output_ref_host
    _ = d_input^
    _ = d_filter^
    _ = d_output^


def main():
    with DeviceContext() as ctx:
        # Check if we're running on an NVIDIA GPU
        if ctx.default_device_info.vendor != Vendor.NVIDIA_GPU:
            print("Skipping cuDNN tests - not running on NVIDIA GPU")
            return

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            in_channels=2,
            out_channels=2,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=1,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=2,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=1,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=1,
            dilation_val=2,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=2,
            pad_val=0,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            stride_val=2,
            dilation_val=2,
            pad_val=1,
        ](ctx=ctx)

        test_conv_transposed_cudnn[
            input_len=550,
            kernel_len=7,
            in_channels=512,
            out_channels=1024,
            stride_val=1,
            dilation_val=1,
            pad_val=3,
        ](ctx=ctx)

    # Test with multiple device contexts consecutively
    print("\n== Testing with multiple device contexts ==")

    # First context - default device (GPU 0)
    print("Creating first device context (default device)...")
    with DeviceContext() as ctx1:
        test_conv_transposed_cudnn[
            input_len=9,
            kernel_len=4,
            in_channels=2,
            out_channels=2,
            stride_val=1,
            dilation_val=1,
            pad_val=0,
        ](ctx=ctx1)

    if DeviceContext.number_of_devices() >= 2:
        # Second context - device 1
        print("Creating second device context (device 1)...")
        with DeviceContext(device_id=1) as ctx2:
            test_conv_transposed_cudnn[
                input_len=9,
                kernel_len=4,
                in_channels=2,
                out_channels=2,
                stride_val=1,
                dilation_val=1,
                pad_val=0,
            ](ctx=ctx2)

        print("Multiple device context test completed successfully!")
