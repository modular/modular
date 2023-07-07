# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import NDBuffer
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_width,
    get_direct_conv_micro_kernel_height,
    get_conv_num_tasks,
    get_conv_num_partitions,
)
from Conv import (
    ConvDirectNHWC,
    Naive2dConvolution,
    pack_filter_rscf_to_frscf,
)
from DType import DType
from IO import print
from Image import ImageData, Image2DLayout, ImageShape
from Index import Index, StaticIntTuple
from Intrinsics import external_call
from List import DimList
from Pointer import DTypePointer
from LLCL import Runtime, OwningOutputChainPtr
from Math import abs, div_ceil, min
from Range import range
from Random import rand
from TargetInfo import simdwidthof

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
    let packed_filter_ptr = DTypePointer[type].alloc(R * S * C * rounded_F)

    let input = NDBuffer[4, DimList.create_unknown[4](), type](
        input_ptr, Index(N, H, W, C), type
    )
    let filter = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C, F), type
    )
    let packed_filter = NDBuffer[5, DimList.create_unknown[5](), type](
        packed_filter_ptr,
        Index(
            div_ceil(F, micro_kernel_width * simd_size),
            R,
            S,
            C,
            micro_kernel_width * simd_size,
        ),
        type,
    )
    let output = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ptr, Index(N, HO, WO, F), type
    )
    let output_ref = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ref_ptr, Index(N, HO, WO, F), type
    )

    @parameter
    if filter_packed:
        pack_filter_rscf_to_frscf[type](filter, packed_filter)

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

    # Test direct conv
    let direct_conv_chain = OwningOutputChainPtr(rt)

    @parameter
    if filter_packed:
        ConvDirectNHWC[
            5,
            DimList.create_unknown[4](),
            DimList.create_unknown[5](),
            DimList.create_unknown[4](),
            type,
            True,
        ].run(
            output, input, packed_filter, conv_shape, direct_conv_chain.borrow()
        )
    else:
        ConvDirectNHWC[
            4,
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            type,
            False,
        ].run(output, input, filter, conv_shape, direct_conv_chain.borrow())
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
        # likely partition in n_ho_wo
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

        # likely partition in F or both
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
            rt,
        )

        # likely to partition C
        test[DType.float32, True](
            1,  # N
            16,  # H
            16,  # W
            256,  # C
            3,  # R
            3,  # S
            256,  # F
            Index(1, 1),  # stride
            Index(1, 1),  # dilation
            Index(0, 0),  # pad_h
            Index(0, 0),  # pad_w
            rt,
        )
