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


# CHECK-LABEL: test_conv1d
fn test[
    type: DType, filter_packed: Bool
](
    N: Int,
    W: Int,
    C: Int,
    S: Int,
    F: Int,
    stride: Int,
    dilation: Int,
    pad_w: StaticIntTuple[2],
    num_groups: Int,
) raises:
    print("== test_conv1d")

    let WO = (W + pad_w[0] + pad_w[1] - dilation * (S - 1) - 1) // stride + 1
    alias HO = 1
    alias H = 1
    alias R = 1

    let conv_shape = ConvShape[1] {
        n: N,
        input_dims: Index(W),
        output_dims: Index(WO),
        filter_dims: Index(S),
        c: C,
        f: F,
        stride: stride,
        dilation: dilation,
        pad_d: Index(0, 0),
        pad_h: Index(0, 0),
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

    # Rounded C and F size for pre-packed filter.
    let micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    let rounded_F = div_ceil(F, micro_kernel_f_size) * micro_kernel_f_size

    let input2d = NDBuffer[4, DimList.create_unknown[4](), type](
        input_ptr, Index(N, H, W, C)
    )
    let filter2d = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C // num_groups, F)
    )
    let packed_filter_shape = pack_conv_filter_shape[type, False](
        filter2d, num_groups
    )
    let packed_filter_ptr = DTypePointer[type].alloc(
        packed_filter_shape.flattened_length()
    )
    let packed_filter = NDBuffer[5, DimList.create_unknown[5](), type](
        packed_filter_ptr,
        packed_filter_shape,
    )
    let output = NDBuffer[3, DimList.create_unknown[3](), type](
        output_ptr, Index(N, WO, F)
    )
    let input1d = NDBuffer[3, DimList.create_unknown[3](), type](
        input_ptr, Index(N, W, C)
    )
    let filter1d = NDBuffer[3, DimList.create_unknown[3](), type](
        filter_ptr, Index(S, C, F)
    )
    let output_ref = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ref_ptr, Index(N, HO, WO, F)
    )

    @parameter
    if filter_packed:
        pack_filter[type](filter2d, packed_filter, num_groups)

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
        ](input2d),
        ImageData[
            DimList.create_unknown[4](),
            type,
            Image2DLayout.RSCF,
        ](filter2d),
        Index(0, 0),
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
            3,
            5,
            3,
            DimList.create_unknown[3](),
            DimList.create_unknown[5](),
            DimList.create_unknown[3](),
            type,
            type,
            type,
            True,
            conv_attr,
            False,
        ].run(
            output,
            input1d,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            3,
            4,
            3,
            DimList.create_unknown[3](),
            DimList.create_unknown[4](),
            DimList.create_unknown[3](),
            type,
            type,
            type,
            False,
            conv_attr,
            False,
        ].run(output, input1d, filter2d, conv_shape)

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    var idx = 0
    for n in range(N):
        for wo in range(WO):
            for f in range(F):
                if not isclose(
                    output_ref.data[idx],
                    output.data[idx],
                    1e-4,  # absolute error tolerance
                    1e-4,  # relative error tolerance
                ):
                    print("Input shape NWC: ", Index(N, W, C))
                    print("filter shape SCF: ", Index(S, C, F))
                    print("filter packed", filter_packed)
                    print("num groups", num_groups)
                    print("Test failed at index: ", Index(n, wo, f))
                    print("Golden value: ", output_ref.data[idx])
                    print("Actual value: ", output.data[idx])
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
    with Runtime() as rt:
        # No packing or padding.
        test[type, False](1, 5, 1, 4, 4, 2, 1, Index(0, 0), 1)
        test[type, False](1, 12, 12, 3, 64, 1, 1, Index(0, 0), 1)
        test[type, False](1, 13, 16, 5, 64, 1, 1, Index(0, 0), 1)
        test[type, False](1, 7, 32, 3, 16, 2, 1, Index(0, 0), 1)

        # Pre-packed test w/o padding.
        test[type, True](1, 17, 16, 5, 64, 3, 1, Index(0, 0), 1)
        test[type, True](5, 12, 8, 3, 64, 2, 1, Index(0, 0), 1)
        test[type, True](1, 7, 11, 3, 192, 3, 1, Index(0, 0), 1)
        test[type, True](1, 7, 5, 5, 256, 2, 1, Index(0, 0), 1)

        # No packing, w/ padding, and F not multiple of simd_size.
        test[type, False](1, 5, 3, 3, 1, 1, 1, Index(1, 1), 1)
        test[type, False](2, 11, 5, 3, 2, 1, 1, Index(1, 1), 1)
        test[type, False](1, 12, 6, 5, 3, 3, 1, Index(2, 2), 1)
        test[type, False](1, 7, 1, 4, 3, 1, 1, Index(2, 1), 1)
        test[type, False](1, 5, 2, 3, 6, 2, 1, Index(1, 1), 1)

        # Pre-packed, F not multiple of simd_size
        test[type, True](1, 5, 2, 3, 7, 1, 1, Index(0, 0), 1)
        test[type, True](1, 7, 2, 3, 42, 2, 1, Index(0, 0), 1)
        test[type, True](1, 23, 17, 3, 90, 1, 1, Index(0, 0), 1)
        test[type, True](1, 11, 2, 5, 7, 1, 1, Index(2, 2), 1)
        test[type, True](1, 9, 2, 3, 42, 2, 1, Index(1, 1), 1)
        test[type, True](1, 7, 17, 5, 90, 2, 1, Index(2, 2), 1)

        # Grouped conv tests
        test[type, True](1, 1, 2, 1, 2, 1, 1, Index(0, 0), 2)
        test[type, True](1, 1, 25, 1, 25, 1, 1, Index(0, 0), 5)
        test[type, True](1, 1, 16, 1, 4, 1, 1, Index(0, 0), 2)
        test[type, True](1, 1, 32, 1, 20, 1, 1, Index(0, 0), 2)
        test[type, True](1, 1, 34, 1, 40, 1, 1, Index(0, 0), 2)
        test[type, True](1, 13, 16, 5, 64, 2, 1, Index(0, 0), 4)
        test[type, True](1, 1, 2, 1, 2, 1, 1, Index(1, 1), 2)
        test[type, True](1, 3, 18, 3, 18, 1, 1, Index(0, 0), 3)
        test[type, True](1, 7, 33, 5, 90, 2, 1, Index(2, 2), 3)
        test[type, True](3, 17, 36, 5, 93, 2, 1, Index(2, 2), 3)
        test[type, True](1, 17, 36, 6, 198, 3, 1, Index(3, 2), 2)

        # Depthwise conv.
        test[type, True](1, 7, 33, 5, 66, 2, 1, Index(2, 2), 33)

        # WavLM and Wav2Vec2 shapes.
        test[type, True](2, 16000, 1, 10, 512, 5, 1, Index(0, 0), 1)
        test[type, True](2, 3199, 512, 3, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 1599, 512, 3, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 799, 512, 3, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 399, 512, 3, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 199, 512, 2, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 99, 512, 2, 512, 2, 1, Index(0, 0), 1)
        test[type, True](2, 49, 1024, 128, 1024, 1, 1, Index(64, 64), 16)
