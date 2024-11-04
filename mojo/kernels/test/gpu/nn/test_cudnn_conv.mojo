# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from random import rand
from sys.info import simdwidthof
from sys import sizeof
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from memory import UnsafePointer
from nn.conv import (
    Naive2dConvolution,
    conv_gpu,
)
from nn.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from utils.index import Index, IndexList
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from testing import assert_equal


fn print_data[type: DType](data: UnsafePointer[Scalar[type]], dim: DimList):
    for i in range(dim.product().get()):
        print(data[i], " ", end="")
    print("")


fn conv2d_gpu_naive_nhwc_rscf[
    input_dim: DimList,
    filter_dim: DimList,
    output_dim: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: UnsafePointer[Scalar[input_type]],
    filter: UnsafePointer[Scalar[filter_type]],
    output: UnsafePointer[Scalar[output_type]],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
):
    # batch index
    var n = BlockIdx.x()
    # output height and width indices
    var h_out = BlockIdx.y()
    var w_out = BlockIdx.z()
    # output channel index
    var c_out = ThreadIdx.x()

    var out_height = output_dim.get[1]()
    var out_width = output_dim.get[2]()
    var N = input_dim.get[0]()
    var H = input_dim.get[1]()
    var W = input_dim.get[2]()
    var C = input_dim.get[3]()  # channel_in
    var KH = filter_dim.get[0]()
    var KW = filter_dim.get[1]()
    var C_out = output_dim.get[3]()  # channel_out
    var pad_h = padding[0]
    var pad_w = padding[1]
    var stride_h = stride[0]
    var stride_w = stride[1]
    var dil_h = dilation[0]
    var dil_w = dilation[1]

    if n < N and h_out < out_height and w_out < out_width and c_out < C_out:
        var sum = Scalar[output_type](0)

        for kh in range(KH):
            for kw in range(KW):
                for c in range(C):
                    var h_in = h_out * stride_h - pad_h + kh * dil_h
                    var w_in = w_out * stride_w - pad_w + kw * dil_w

                    if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        var input_val = input[
                            n * H * W * C + h_in * W * C + w_in * C + c
                        ].cast[output_type]()
                        var filter_val = filter[
                            kh * KW * C + kw * C + c * C_out + c_out
                        ].cast[output_type]()
                        sum += input_val * filter_val

        output[
            n * out_height * out_width * C_out
            + h_out * out_width * C_out
            + w_out * C_out
            + c_out
        ] = sum


# input: NHWC
# filer: RSCF
fn test_conv[
    input_dim: DimList,
    filter_dim: DimList,
    output_dim: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    stride_dim: IndexList[2],
    dilation_dim: IndexList[2],
    pad_dim: IndexList[2],
](ctx: DeviceContext) raises:
    print("== test_cudnn_conv_gpu")

    var input_dim_flattened = input_dim.product().get()
    var filter_dim_flattened = filter_dim.product().get()
    var output_dim_flattened = output_dim.product().get()

    var input_data_host = UnsafePointer[Scalar[input_type]].alloc(
        input_dim_flattened
    )
    var filter_data_host = UnsafePointer[Scalar[filter_type]].alloc(
        filter_dim_flattened
    )
    var output_data_host = UnsafePointer[Scalar[output_type]].alloc(
        output_dim_flattened
    )
    var output_ref_host = UnsafePointer[Scalar[output_type]].alloc(
        output_dim_flattened
    )

    for i in range(input_dim_flattened):
        input_data_host[i] = Scalar[input_type](i + 1)
    for i in range(filter_dim_flattened):
        filter_data_host[i] = Scalar[filter_type](1)

    var input_dev = ctx.create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.create_buffer[filter_type](filter_dim_flattened)
    var output_dev = ctx.create_buffer[output_type](output_dim_flattened)
    var output_ref_dev = ctx.create_buffer[output_type](output_dim_flattened)

    ctx.enqueue_copy_to_device(input_dev, input_data_host)
    ctx.enqueue_copy_to_device(filter_dev, filter_data_host)

    var conv_naive = ctx.compile_function[
        conv2d_gpu_naive_nhwc_rscf[
            input_dim,
            filter_dim,
            output_dim,
            input_type,
            filter_type,
            output_type,
        ]
    ]()

    ctx.enqueue_function(
        conv_naive,
        input_dev.ptr,
        filter_dev.ptr,
        output_ref_dev.ptr,
        stride_dim,
        dilation_dim,
        pad_dim,
        grid_dim=(input_dim.get[0](), output_dim.get[1](), output_dim.get[2]()),
        block_dim=output_dim.get[3](),
    )

    ctx.enqueue_copy_from_device(output_ref_host, output_ref_dev)

    conv_gpu[
        input_dim,
        DimList(
            filter_dim.get[3](),
            filter_dim.get[2](),
            filter_dim.get[0](),
            filter_dim.get[1](),
        ),
        output_dim,
        input_type,
        filter_type,
        output_type,
    ](
        input_dev.ptr,
        filter_dev.ptr,
        output_dev.ptr,
        stride_dim,
        dilation_dim,
        pad_dim,
        1,
        ctx,
    )

    ctx.enqueue_copy_from_device(output_data_host, output_dev)

    # verifying results
    for x in range(output_dim_flattened):
        assert_equal(output_data_host[x], output_ref_host[x])


def main():
    with DeviceContext() as ctx:
        test_conv[
            DimList(1, 5, 5, 1),  # input  (NHWC)
            DimList(
                3, 3, 1, 1
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DimList(1, 3, 3, 1),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv[
            DimList(1, 32, 32, 3),  # input  (NHWC)
            DimList(
                5, 5, 3, 16
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DimList(1, 32, 32, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](2, 2),  # pad
        ](ctx)

        test_conv[
            DimList(2, 28, 28, 1),  # input  (NHWC)
            DimList(
                3, 3, 1, 8
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DimList(2, 13, 13, 8),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](2, 2),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)
