# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose
from random import rand
from sys import sizeof
from sys.info import num_physical_cores, simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from memory import UnsafePointer
from nn.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    conv_cudnn,
    conv_gpu,
    conv_nhwc_direct,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv_utils import (
    ConvShape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)
from testing import assert_almost_equal, assert_equal

from utils.index import Index, IndexList


fn print_data[type: DType](data: UnsafePointer[Scalar[type]], dim: DimList):
    for i in range(dim.product().get()):
        print(data[i], " ", end="")
    print("")


# input: NHWC
# filer: RSCF
fn test_conv_cudnn[
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

    var input_dev = ctx.enqueue_create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var output_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )
    var output_ref_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )

    ctx.enqueue_copy(input_dev, input_data_host)
    ctx.enqueue_copy(filter_dev, filter_data_host)

    conv_gpu[
        4,
        4,
        input_dim,
        filter_dim,
        output_dim,
        input_type,
        filter_type,
        output_type,
    ](
        input_dev.unsafe_ptr(),
        filter_dev.unsafe_ptr(),
        output_ref_dev.unsafe_ptr(),
        stride_dim,
        dilation_dim,
        pad_dim,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_ref_host, output_ref_dev)

    conv_cudnn[
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
        input_dev.unsafe_ptr(),
        filter_dev.unsafe_ptr(),
        output_dev.unsafe_ptr(),
        stride_dim,
        dilation_dim,
        pad_dim,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_data_host, output_dev)

    # verifying results
    for x in range(output_dim_flattened):
        assert_equal(output_data_host[x], output_ref_host[x])

    input_data_host.free()
    filter_data_host.free()
    output_data_host.free()
    output_ref_host.free()


# input: NHWC
# filer: RSCF
fn test_conv_gpu[
    input_dim: DimList,
    filter_dim: DimList,
    type: DType,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[2],
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print("== test_conv_gpu")

    alias filter_packed = False
    alias simd_size: Int = simdwidthof[DType.float32]()
    alias N = input_dim.get[0]()
    alias H = input_dim.get[1]()
    alias W = input_dim.get[2]()
    alias C = input_dim.get[3]()
    alias R = filter_dim.get[0]()
    alias S = filter_dim.get[1]()
    alias F = filter_dim.get[3]()
    alias pad_h = IndexList[2](pad[0], pad[0])
    alias pad_w = IndexList[2](pad[1], pad[1])

    alias HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[
        0
    ] + 1
    alias WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[
        1
    ] + 1
    alias output_dim = DimList(N, HO, WO, F)

    var input_dim_flattened = input_dim.product().get()
    var filter_dim_flattened = filter_dim.product().get()
    var output_dim_flattened = output_dim.product().get()

    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Index(H, W),
        output_dims=Index(HO, WO),
        filter_dims=Index(R, S),
        c=C,
        f=F,
        stride=stride,
        dilation=dilation,
        pad_d=Index(0, 0),
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )

    var input_ptr = UnsafePointer[Scalar[type]].alloc(input_dim_flattened)
    var filter_ptr = UnsafePointer[Scalar[type]].alloc(filter_dim_flattened)
    var output_cpu_ptr = UnsafePointer[Scalar[type]].alloc(output_dim_flattened)
    var output_gpu_ptr = UnsafePointer[Scalar[type]].alloc(output_dim_flattened)
    var output_ref_ptr = UnsafePointer[Scalar[type]].alloc(output_dim_flattened)

    rand[type](input_ptr, input_dim_flattened)
    rand[type](filter_ptr, filter_dim_flattened)

    var input = NDBuffer[type, 4](input_ptr, IndexList[4](N, H, W, C))
    var filter = NDBuffer[type, 4](
        filter_ptr, IndexList[4](R, S, C // num_groups, F)
    )
    var output_cpu = NDBuffer[type, 4](
        output_cpu_ptr, IndexList[4](N, HO, WO, F)
    )
    var output_gpu = NDBuffer[type, 4](
        output_gpu_ptr, IndexList[4](N, HO, WO, F)
    )
    var output_ref = NDBuffer[type, 4](
        output_ref_ptr, IndexList[4](N, HO, WO, F)
    )

    var input_dev = ctx.enqueue_create_buffer[type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[type](filter_dim_flattened)
    var output_dev = ctx.enqueue_create_buffer[type](output_dim_flattened)

    var input_buf = NDBuffer[type, 4, _, input_dim](
        input_dev.unsafe_ptr(), input_dim
    )
    var filter_buf = NDBuffer[type, 4, _, filter_dim](
        filter_dev.unsafe_ptr(), filter_dim
    )
    var output_buf = NDBuffer[type, 4, _, output_dim](
        output_dev.unsafe_ptr(), output_dim
    )

    # Reference: naive conv
    Naive2dConvolution[
        type,  # Data type.
        type,
        type,
    ].run(
        output_ref_ptr,
        input_ptr,
        filter_ptr,
        Index(N, 1, HO, WO, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C // num_groups, F),
        Index(0, 0),  #  pad_d
        pad_h,
        pad_w,
        (1, stride[0], stride[1]),
        (1, dilation[0], dilation[1]),
        num_groups,
    )

    # Test direct conv
    alias conv_attr = ConvInfoStatic[2]()

    ConvDirectNHWC[
        4,
        4,
        4,
        _,
        _,
        _,
        DimList.create_unknown[4](),
        DimList.create_unknown[4](),
        DimList.create_unknown[4](),
        type,
        type,
        type,
        False,
        conv_attr,
    ].run(output_cpu, input, filter, conv_shape)

    ctx.enqueue_copy(input_dev, input_ptr)
    ctx.enqueue_copy(filter_dev, filter_ptr)

    # Run with convolution with epilogue fusion:
    @parameter
    @always_inline
    @__copy_capture(output_buf)
    fn output_fn[
        _type: DType, _rank: Int, _width: Int
    ](coords: IndexList[_rank], val: SIMD[_type, _width]) capturing:
        output_buf.store[width=_width](
            rebind[IndexList[output_buf.rank]](coords),
            rebind[SIMD[output_buf.type, _width]](val) + 1,
        )

    conv_gpu[
        4,
        4,
        input_dim,
        filter_dim,
        output_dim,
        type,
        type,
        type,
        maybe_epilogue_func=output_fn,
    ](
        input_buf,
        filter_buf,
        output_buf,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    ctx.enqueue_copy(output_gpu_ptr, output_dev)

    # Account for epilogue which adds one but is not in reference impls.
    for i in range(len(output_ref)):
        output_ref.data[i] += 1
        output_cpu.data[i] += 1

    # verifying results
    for x in range(output_dim_flattened):
        assert_equal(output_ref_ptr[x], output_cpu_ptr[x])
        assert_almost_equal(output_ref_ptr[x], output_gpu_ptr[x], rtol=0.01)

    input_ptr.free()
    filter_ptr.free()

    output_cpu_ptr.free()
    output_gpu_ptr.free()
    output_ref_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_conv_gpu[
            DimList(1, 64, 64, 32),  # input  (NHWC)
            DimList(
                3, 3, 32, 64
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](2, 2),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(1, 5, 5, 2),  # input  (NHWC)
            DimList(
                3, 3, 2, 2
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](2, 2),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(1, 5, 5, 1),  # input  (NHWC)
            DimList(
                3, 3, 1, 1
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(1, 4, 4, 3),  # input  (NHWC)
            DimList(
                2, 2, 3, 1
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(1, 5, 5, 3),  # input  (NHWC)
            DimList(
                3, 3, 3, 1
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(5, 7, 7, 8),  # input  (NHWC)
            DimList(
                3, 3, 8, 64
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(1, 7, 7, 7),  # input  (NHWC)
            DimList(
                3, 3, 7, 256
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](3, 3),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)

        test_conv_gpu[
            DimList(2, 28, 28, 1),  # input  (NHWC)
            DimList(
                3, 3, 1, 8
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DType.float32,
            IndexList[2](3, 3),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](2, 2),  # pad
        ](ctx)

        test_conv_cudnn[
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

        test_conv_cudnn[
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
