# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Bool import Bool
from Buffer import NDBuffer
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_width,
)
from Conv import (
    ConvDirectNHWC,
    Naive2dConvolution,
    pack_filter_rscf_to_cfrscf,
)
from DType import DType
from IO import print
from Image import ImageData, Image2DLayout, ImageShape
from Index import Index, StaticIntTuple
from Intrinsics import external_call
from List import DimList
from Pointer import DTypePointer
from LLCL import Runtime, OwningOutputChainPtr
from Math import abs, div_ceil
from Range import range
from Random import rand
from TargetInfo import dtype_simd_width

alias simd_size: Int = dtype_simd_width[DType.float32]()
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
    rt: Runtime,
):
    print("== test_direct_conv")

    # fmt: off
    let HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    let WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    # fmt: on

    let conv_shape = ConvShape {
        n: N,
        h: H,
        w: W,
        c: C,
        out_h: HO,
        out_w: WO,
        f: F,
        r: R,
        s: S,
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }

    let input_ptr = DTypePointer[type].alloc(N * H * W * C)
    let filter_ptr = DTypePointer[type].alloc(R * S * C * F)
    let output_ptr = DTypePointer[type].alloc(N * HO * WO * F)
    let output_ref_ptr = DTypePointer[type].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    let cf_tile_size = get_conv_tile_shape[
        type,
        get_direct_conv_micro_kernel_width(),
    ](conv_shape)

    # Rounded C and F size for pre-packed filter.
    let micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    let rounded_F = div_ceil(F, micro_kernel_f_size) * micro_kernel_f_size
    let packed_filter_ptr = DTypePointer[type].alloc(R * S * C * rounded_F)

    @parameter
    if filter_packed:
        pack_filter_rscf_to_cfrscf[
            get_direct_conv_micro_kernel_width(),
            simd_size,
            type,
        ](conv_shape, cf_tile_size[0], filter_ptr, packed_filter_ptr)

    let input = NDBuffer[4, DimList.create_unknown[4](), type](
        input_ptr, Index(N, H, W, C), type
    )
    let filter = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C, F), type
    )
    let packed_filter = NDBuffer[4, DimList.create_unknown[4](), type](
        packed_filter_ptr, Index(R, S, C, rounded_F), type
    )
    let output = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ptr, Index(N, HO, WO, F), type
    )
    let output_ref = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ref_ptr, Index(N, HO, WO, F), type
    )

    # Reference: naive conv
    Naive2dConvolution[
        DimList.create_unknown[4](),  # Output Shape.
        DimList.create_unknown[4](),  # Filter Shape,
        DimList.create_unknown[4](),  # Input Shape
        type,  # Data type.
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
    )

    let filter_test = packed_filter if filter_packed else filter

    # Test direct conv
    let direct_conv_chain = OwningOutputChainPtr(rt)
    ConvDirectNHWC[
        DimList.create_unknown[4](),
        DimList.create_unknown[4](),
        DimList.create_unknown[4](),
        type,
        filter_packed,
    ].run(output, input, filter_test, conv_shape, direct_conv_chain.borrow())
    direct_conv_chain.wait()

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    for n in range(N):
        for ho in range(HO):
            for wo in range(WO):
                for f in range(F):
                    if (
                        abs(output_ref[n, ho, wo, f] - output[n, ho, wo, f])
                        > abs(output_ref[n, ho, wo, f]) * 1e-5
                    ):
                        print("Input shape NHWC: ", Index(N, H, W, C))
                        print("filter shape RSCF: ", Index(R, S, C, F))
                        print("filter packed", filter_packed)
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


fn main():
    """It only includes shapes where F is multiple simd_size."""
    with Runtime() as rt:
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
            rt,
        )

        test[DType.float32, False](
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
            rt,
        )

        # Pre-packed test
        # Avoid using dispatch functions for now because pre-packed version
        # has more restrictions for F.

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
            rt,
        )
