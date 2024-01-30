# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import abs, div_ceil, isclose, min
from random import rand, seed
from sys import external_call
from sys.info import simdwidthof

from NN.Conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_conv_filter_shape,
    pack_filter,
)
from NN.ConvUtils import (
    ConvShape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)
from NN.Image import Image2DLayout, ImageData, ImageShape
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from runtime.llcl import Runtime

from utils.index import Index, StaticIntTuple
from utils.list import DimList

alias simd_size: Int = simdwidthof[DType.float32]()
alias type = DType.float32


# CHECK-LABEL: test_direct_conv
fn test[
    type: DType, filter_packed: Bool
](
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    F: Int,
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    num_groups: Int,
    rt: Runtime,
) raises:
    print("== test_direct_conv")

    # fmt: off
    let HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    let WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    # fmt: on

    let conv_shape = ConvShape[2] {
        n: N,
        input_dims: Index(H, W),
        output_dims: Index(HO, WO),
        filter_dims: Index(R, S),
        c: C,
        f: F,
        stride: stride,
        dilation: dilation,
        pad_d: Index(0, 0),
        pad_h: pad_h,
        pad_w: pad_w,
        num_groups: num_groups,
    }

    let input_ptr = DTypePointer[type].alloc(N * H * W * C)
    let filter_ptr = DTypePointer[type].alloc(R * S * C * F)
    let output_ptr = DTypePointer[type].alloc(N * HO * WO * F)
    let output_ref_ptr = DTypePointer[type].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    let num_threads = rt.parallelism_level()
    let num_tasks = get_conv_num_tasks(num_threads, conv_shape)
    let num_partitions = get_conv_num_partitions[
        micro_kernel_height, micro_kernel_width * simd_size
    ](num_tasks, conv_shape)

    # Rounded C and F size for pre-packed filter.
    let micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    let rounded_F = div_ceil(F, micro_kernel_f_size) * micro_kernel_f_size

    let input = NDBuffer[4, DimList.create_unknown[4](), type](
        input_ptr, Index(N, H, W, C)
    )
    let filter = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C // num_groups, F)
    )
    let packed_filter_shape = pack_conv_filter_shape[type, False](
        filter, num_groups
    )
    let packed_filter_ptr = DTypePointer[type].alloc(
        packed_filter_shape.flattened_length()
    )
    let packed_filter = NDBuffer[5, DimList.create_unknown[5](), type](
        packed_filter_ptr,
        packed_filter_shape,
    )
    let output = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ptr, Index(N, HO, WO, F)
    )
    let output_ref = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ref_ptr, Index(N, HO, WO, F)
    )

    @parameter
    if filter_packed:
        pack_filter[type](filter, packed_filter, num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        DimList.create_unknown[4](),  # Output Shape.
        DimList.create_unknown[4](),  # Filter Shape,
        DimList.create_unknown[4](),  # Input Shape
        type,  # Data type.
        type,
        type,
        Image2DLayout.NHWC,  # Data Layout.
        Image2DLayout.RSCF,  # Filter Layout.
    ].run(
        ImageData[
            DimList.create_unknown[4](),
            type,
            Image2DLayout.NHWC,
        ](output_ref),
        ImageData[
            DimList.create_unknown[4](),
            type,
            Image2DLayout.NHWC,
        ](input),
        ImageData[
            DimList.create_unknown[4](),
            type,
            Image2DLayout.RSCF,
        ](filter),
        pad_h,
        pad_w,
        stride,
        dilation,
        num_groups,
    )

    # Test direct conv
    alias conv_attr = ConvInfoStatic.create_unknown()

    @parameter
    if filter_packed:
        ConvDirectNHWC[
            4,
            5,
            4,
            DimList.create_unknown[4](),
            DimList.create_unknown[5](),
            DimList.create_unknown[4](),
            type,
            type,
            type,
            True,
            conv_attr,
            False,
        ].run(
            output,
            input,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            4,
            4,
            4,
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            type,
            type,
            type,
            False,
            conv_attr,
            False,
        ].run(output, input, filter, conv_shape)

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    for n in range(N):
        for ho in range(HO):
            for wo in range(WO):
                for f in range(F):
                    if not isclose(
                        output_ref[n, ho, wo, f],
                        output[n, ho, wo, f],
                        1e-4,  # absolute error tolerance
                        1e-4,  # relative error tolerance
                    ):
                        print("Input shape NHWC: ", Index(N, H, W, C))
                        print("filter shape RSCF: ", Index(R, S, C, F))
                        print("filter packed", filter_packed)
                        print("num groups", num_groups)
                        print("Test failed at index: ", Index(n, ho, wo, f))
                        print("Golden value: ", output_ref[n, ho, wo, f])
                        print("Actual value: ", output[n, ho, wo, f])
                        output_ptr.free()
                        output_ref_ptr.free()
                        return

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


fn main() raises:
    """It only includes shapes where F is multiple simd_size."""
    with Runtime() as rt:
        # No packing or padding.
        test[DType.float32, False](
            1,  # N
            6,  # H
            5,  # W
            1,  # C
            3,  # R
            4,  # S
            4,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            12,  # H
            12,  # W
            12,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            13,  # H
            13,  # W
            16,  # C
            5,  # R
            5,  # S
            64,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            7,  # H
            7,  # W
            32,  # C
            3,  # R
            3,  # S
            16,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            17,  # H
            17,  # W
            16,  # C
            5,  # R
            5,  # S
            32,  # F
            Index(3, 3),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            5,  # N
            7,  # H
            7,  # W
            8,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        # likely partition in F or both
        test[DType.float32, False](
            1,  # N
            7,  # H
            7,  # W
            7,  # C
            3,  # R
            3,  # S
            256,  # F
            Index(3, 3),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            7,  # H
            7,  # W
            5,  # C
            5,  # R
            5,  # S
            288,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        # Pre-packed test w/o padding.

        test[DType.float32, True](
            1,  # N
            12,  # H
            12,  # W
            12,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            13,  # H
            13,  # W
            16,  # C
            5,  # R
            5,  # S
            64,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            7,  # W
            32,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            17,  # H
            17,  # W
            16,  # C
            5,  # R
            5,  # S
            64,  # F
            Index(3, 3),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            5,  # N
            12,  # H
            12,  # W
            8,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            7,  # W
            11,  # C
            3,  # R
            3,  # S
            192,  # F
            Index(3, 3),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            7,  # W
            5,  # C
            5,  # R
            5,  # S
            256,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        # No packing, w/ padding, and F not multiple of simd_size.

        test[DType.float32, False](
            1,  # N
            5,  # H
            5,  # W
            3,  # C
            3,  # R
            3,  # S
            1,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            2,  # N
            12,  # H
            11,  # W
            5,  # C
            4,  # R
            3,  # S
            2,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            8,  # H
            12,  # W
            6,  # C
            2,  # R
            5,  # S
            3,  # F
            Index(1, 3),  # stride
            Index(1, 1),  # dilation
            Index(1, 0),  # pad_h
            Index(2, 2),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            9,  # H
            7,  # W
            1,  # C
            5,  # R
            4,  # S
            3,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(2, 2),  # pad_h
            Index(2, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, False](
            1,  # N
            10,  # H
            5,  # W
            2,  # C
            4,  # R
            3,  # S
            6,  # F
            Index(3, 2),  # stride
            Index(1, 1),  # dilation
            Index(2, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        # Pre-packed, F not multiple of simd_size

        test[DType.float32, True](
            1,  # N
            5,  # H
            5,  # W
            2,  # C
            3,  # R
            3,  # S
            7,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            7,  # W
            2,  # C
            3,  # R
            3,  # S
            42,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            23,  # H
            23,  # W
            17,  # C
            3,  # R
            3,  # S
            90,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            5,  # H
            11,  # W
            2,  # C
            3,  # R
            5,  # S
            7,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(2, 2),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            9,  # W
            2,  # C
            3,  # R
            3,  # S
            42,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            11,  # H
            7,  # W
            17,  # C
            3,  # R
            5,  # S
            90,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(2, 2),  # pad_w
            1,  # num_groups
            rt,
        )
        # Top resnet shapes, all pre-packed w/ padding.

        test[DType.float32, True](
            1,  # N
            224,  # H
            224,  # W
            3,  # C
            7,  # R
            7,  # S
            64,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(3, 3),  # pad_h
            Index(3, 3),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            56,  # H
            56,  # W
            64,  # C
            3,  # R
            3,  # S
            64,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            56,  # H
            56,  # W
            128,  # C
            3,  # R
            3,  # S
            128,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            28,  # H
            28,  # W
            256,  # C
            3,  # R
            3,  # S
            256,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            14,  # H
            14,  # W
            256,  # C
            3,  # R
            3,  # S
            256,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            14,  # H
            14,  # W
            3,  # C
            3,  # R
            3,  # S
            16,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            7,  # H
            7,  # W
            512,  # C
            3,  # R
            3,  # S
            512,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            19,  # N
            7,  # H
            7,  # W
            1,  # C
            3,  # R
            3,  # S
            16,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            13,  # N
            14,  # H
            14,  # W
            2,  # C
            3,  # R
            3,  # S
            32,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            1,  # num_groups
            rt,
        )

        # MaskRCNN shapes.

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            256,  # C
            3,  # R
            3,  # S
            384,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            288,  # C
            3,  # R
            3,  # S
            320,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            256,  # C
            3,  # R
            3,  # S
            288,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            256,  # C
            3,  # R
            3,  # S
            384,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            288,  # C
            3,  # R
            3,  # S
            320,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            2,  # N
            19,  # H
            19,  # W
            256,  # C
            3,  # R
            3,  # S
            288,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            129,  # H
            129,  # W
            320,  # C
            3,  # R
            3,  # S
            384,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            129,  # H
            129,  # W
            256,  # C
            3,  # R
            3,  # S
            384,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1025,  # H
            1025,  # W
            3,  # C
            3,  # R
            3,  # S
            32,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            1,  # num_groups
            rt,
        )

        # grouped conv tests
        # focus on C, F, and num_groups since grouped conv is independent of spatial dims
        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            2,  # C
            1,  # R
            1,  # S
            2,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            2,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            25,  # C
            1,  # R
            1,  # S
            25,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            5,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            16,  # C
            1,  # R
            1,  # S
            4,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            2,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            32,  # C
            1,  # R
            1,  # S
            20,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            2,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            34,  # C
            1,  # R
            1,  # S
            40,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            2,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            13,  # H
            13,  # W
            16,  # C
            5,  # R
            5,  # S
            64,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            4,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            1,  # H
            1,  # W
            2,  # C
            1,  # R
            1,  # S
            2,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(1, 1),  # pad_w
            2,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            3,  # H
            3,  # W
            18,  # C
            3,  # R
            3,  # S
            18,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            3,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            11,  # H
            7,  # W
            33,  # C
            3,  # R
            5,  # S
            90,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(2, 2),  # pad_w
            3,  # num_groups
            rt,
        )

        test[DType.float32, True](
            3,  # N
            11,  # H
            17,  # W
            36,  # C
            3,  # R
            5,  # S
            93,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(2, 2),  # pad_w
            3,  # num_groups
            rt,
        )

        test[DType.float32, True](
            1,  # N
            11,  # H
            17,  # W
            36,  # C
            2,  # R
            6,  # S
            198,  # F
            Index(2, 3),  # stride
            Index(1, 1),  # dilation
            Index(1, 0),  # pad_h
            Index(3, 2),  # pad_w
            2,  # num_groups
            rt,
        )

        # depthwise conv
        test[DType.float32, True](
            1,  # N
            11,  # H
            7,  # W
            33,  # C
            3,  # R
            5,  # S
            66,  # F
            Index(2, 2),  # stride
            Index(1, 1),  # dilation
            Index(1, 1),  # pad_h
            Index(2, 2),  # pad_w
            33,  # num_groups
            rt,
        )

        # 1D edge case
        test[DType.float32, True](
            2,  # N
            1,  # H
            49,  # W
            1024,  # C
            1,  # R
            128,  # S
            1024,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(64, 64),  # pad_w
            64,  # num_groups
            rt,
        )
