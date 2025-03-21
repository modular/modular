# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv, isclose
from random import rand
from sys.info import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from utils.index import Index, IndexList

alias simd_size: Int = simdwidthof[DType.float32]()


# CHECK-LABEL: test_conv3d
fn test[
    type: DType, filter_packed: Bool
](
    N: Int,
    DHW: IndexList[3],
    C: Int,
    QRS: IndexList[3],
    F: Int,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    print("== test_conv3d")

    var D = DHW[0]
    var H = DHW[1]
    var W = DHW[2]

    var Q = QRS[0]
    var R = QRS[1]
    var S = QRS[2]

    # fmt: off
    var DO = (D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1) // stride[0] + 1
    var HO = (H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1) // stride[1] + 1
    var WO = (W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1) // stride[2] + 1
    # fmt: on

    var conv_shape = ConvShape[3](
        n=N,
        input_dims=DHW,
        output_dims=Index(DO, HO, WO),
        filter_dims=QRS,
        c=C,
        f=F,
        stride=stride,
        dilation=dilation,
        pad_d=pad_d,
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )

    var C_per_group = C // num_groups

    var input_ptr = UnsafePointer[Scalar[type]].alloc(N * D * H * W * C)
    var filter_ptr = UnsafePointer[Scalar[type]].alloc(
        Q * R * S * C_per_group * F
    )
    var output_ptr = UnsafePointer[Scalar[type]].alloc(N * DO * HO * WO * F)
    var output_ref_ptr = UnsafePointer[Scalar[type]].alloc(N * DO * HO * WO * F)

    rand[type](input_ptr, N * D * H * W * C)
    rand[type](filter_ptr, Q * R * S * C_per_group * F)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    var micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    # Buffers for direct conv.
    var input = NDBuffer[type, 5](input_ptr, Index(N, D, H, W, C))
    var filter = NDBuffer[type, 5](filter_ptr, Index(Q, R, S, C_per_group, F))
    var packed_filter_shape = pack_conv_filter_shape[False](filter, num_groups)

    var packed_filter_ptr = UnsafePointer[Scalar[type]].alloc(
        packed_filter_shape.flattened_length()
    )
    var packed_filter = NDBuffer[type, 6](
        packed_filter_ptr,
        packed_filter_shape,
    )
    var output = NDBuffer[type, 5](output_ptr, Index(N, DO, HO, WO, F))

    @parameter
    if filter_packed:
        pack_filter(filter, packed_filter, num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        type,
        type,
        type,
    ].run(
        output_ref_ptr,
        input_ptr,
        filter_ptr,
        Index(N, DO, HO, WO, F),
        Index(N, D, H, W, C),
        Index(Q, R, S, C // num_groups, F),
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        num_groups,
    )

    # Test direct conv
    alias conv_attr = ConvInfoStatic[3]()

    @parameter
    if filter_packed:
        ConvDirectNHWC[
            5,
            6,
            5,
            _,
            _,
            _,
            DimList.create_unknown[5](),
            DimList.create_unknown[6](),
            DimList.create_unknown[5](),
            type,
            type,
            type,
            True,
            conv_attr,
        ].run(output, input, packed_filter, conv_shape)
    else:
        ConvDirectNHWC[
            5,
            5,
            5,
            _,
            _,
            _,
            DimList.create_unknown[5](),
            DimList.create_unknown[5](),
            DimList.create_unknown[5](),
            type,
            type,
            type,
            False,
            conv_attr,
        ].run(output, input, filter, conv_shape)

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    var idx = 0
    for n in range(N):
        for do in range(DO):
            for ho in range(HO):
                for wo in range(WO):
                    for f in range(F):
                        if not isclose(
                            output_ref_ptr[idx],
                            output_ptr[idx],
                            atol=1e-4,  # absolute error tolerance
                            rtol=1e-4,  # relative error tolerance
                        ):
                            print("Input shape: ", Index(N, D, H, W, C))
                            print("filter shape: ", Index(Q, R, S, C, F))
                            print("filter packed", filter_packed)
                            print("num groups", num_groups)
                            print(
                                "Test failed at index: ",
                                Index(n, do, ho, wo, f),
                            )
                            print("Golden value: ", output_ref_ptr[idx])
                            print("Actual value: ", output_ptr[idx])
                            output_ptr.free()
                            output_ref_ptr.free()
                            return
                        idx += 1

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


fn main() raises:
    alias type = DType.float32

    # w/o packing, w/o padding.
    test[type, False](
        1,  # batch size
        Index(2, 4, 5),  # input shape
        4,  # C
        Index(1, 2, 3),  # filter shape
        3,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )
    test[type, False](
        1,  # batch size
        Index(9, 8, 5),  # input shape
        1,  # C
        Index(2, 2, 3),  # filter shape
        32,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # w/o packing, w/ padding.
    test[type, False](
        1,  # batch size
        Index(5, 7, 6),  # input shape
        7,  # C
        Index(3, 4, 3),  # filter shape
        24,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )
    test[type, False](
        1,  # batch size
        Index(10, 11, 6),  # input shape
        2,  # C
        Index(3, 4, 3),  # filter shape
        31,  # F
        Index(2, 3, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(2, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # w/ packing, w/o padding.
    test[type, True](
        1,  # batch size
        Index(11, 13, 17),  # input shape
        9,  # C
        Index(7, 5, 3),  # filter shape
        3,  # F
        Index(1, 2, 4),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )
    test[type, True](
        1,  # batch size
        Index(13, 9, 7),  # input shape
        4,  # C
        Index(4, 7, 3),  # filter shape
        17,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(0, 0),  # pad_d
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # w/ packing, w/ padding.
    test[type, True](
        1,  # batch size
        Index(5, 5, 5),  # input shape
        4,  # C
        Index(3, 3, 3),  # filter shape
        64,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )
    test[type, True](
        1,  # batch size
        Index(11, 9, 14),  # input shape
        4,  # C
        Index(4, 7, 3),  # filter shape
        3,  # F
        Index(2, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(2, 1),  # pad_d
        Index(3, 3),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # 3D-UNet shapes.
    # Leave large shapes in comments to save time for CI.
    test[type, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[type, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(2, 2, 2),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[type, True](
        1,  # batch size
        Index(4, 4, 4),  # input shape
        320,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[type, True](
        1,  # batch size
        Index(8, 8, 8),  # input shape
        640,  # C
        Index(3, 3, 3),  # filter shape
        320,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        Index(1, 1),  # pad_d
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # test[type, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     1,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     320,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     512,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(16, 16, 16),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     256,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     256,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(32, 32, 32),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     128,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     128,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(64, 64, 64),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     64,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     64,  # C
    #     Index(3, 3, 3),  # filter shape
    #     32,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(1, 1),  # pad_d
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[type, True](
    #     1,  # batch size
    #     Index(128, 128, 128),  # input shape
    #     32,  # C
    #     Index(3, 3, 3),  # filter shape
    #     3,  # F
    #     Index(1, 1, 1),  # stride
    #     Index(1, 1, 1),  # dilation
    #     Index(0, 0),  # pad_d
    #     Index(0, 0),  # pad_h
    #     Index(0, 0),  # pad_w
    #     1,  # num_groups
    # )
