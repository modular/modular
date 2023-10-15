# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import abs, div_ceil, min
from random import rand
from sys import external_call
from sys.info import simdwidthof

from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from nn.conv import (
    ConvDirectNHWC,
    ConvIm2ColNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_filter,
)
from nn.conv_utils import (
    ConvAlgorithm,
    ConvShape,
    get_conv_a_row_size,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_pack_inner_size,
    get_conv_tile_shape,
    get_conv_tile_size,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)
from nn.image import Image2DLayout, ImageData, ImageShape
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.index import Index, StaticIntTuple
from utils.list import DimList

alias simd_size: Int = simdwidthof[DType.float32]()
alias type = DType.float32


# CHECK-LABEL: test_conv
fn test[
    type: DType, algorithm: ConvAlgorithm, filter_packed: Bool
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
    print("== test_conv")

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
        num_groups: 1,
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
        input_ptr, Index(N, H, W, C)
    )
    let filter = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C, F)
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
    )
    let output = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ptr, Index(N, HO, WO, F)
    )
    let output_ref = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ref_ptr, Index(N, HO, WO, F)
    )

    @parameter
    if filter_packed:
        pack_filter[type](filter, packed_filter, conv_shape.num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        DimList.create_unknown[4](),  # Output Shape.
        DimList.create_unknown[4](),  # Filter Shape,
        DimList.create_unknown[4](),  # Input Shape
        type,  # Input data type.
        type,  # Filter data type.
        type,  # Output Data type.
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
        conv_shape.num_groups,
    )

    # Test direct conv
    let conv_chain = OwningOutputChainPtr(rt)

    if algorithm == ConvAlgorithm.Direct:
        alias conv_attr = ConvInfoStatic.create_unknown()

        @parameter
        if filter_packed:
            ConvDirectNHWC[
                5,
                DimList.create_unknown[4](),
                DimList.create_unknown[5](),
                DimList.create_unknown[4](),
                type,
                type,
                type,
                True,
                conv_attr,
                False,
            ].run(output, input, packed_filter, conv_shape, conv_chain.borrow())
        else:
            ConvDirectNHWC[
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
            ].run(output, input, filter, conv_shape, conv_chain.borrow())
    elif algorithm == ConvAlgorithm.Im2Col:
        ConvIm2ColNHWC[
            # Input Shape
            DimList.create_unknown[4](),
            # Filter Shape
            DimList.create_unknown[4](),
            # Output Shape
            DimList.create_unknown[4](),
            # Packed Shape
            DimList.create_unknown[3](),
            type,
            simd_size,
            get_conv_a_row_size(),
            get_conv_pack_inner_size() * simd_size,
            get_conv_tile_size[type](),
            # Filter layout.
            Image2DLayout.RSCF,
            # elementwise_epilogue_enabled
            False,
        ].run(
            output,
            input,
            filter,
            conv_shape,
            conv_chain.borrow(),
        )
    conv_chain.wait()

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
                        print("algorithm", algorithm.value)
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


fn test_shapes[algorithm: ConvAlgorithm](rt: Runtime):
    """This test uses shapes in Models/mask_rcnn_inception_resnetv2_backbone.yaml.
    The original shapes are too large for unit test so the batch size is reduced
    to 2. The shapes with strides > 1 are chosen for #17704.
    """
    alias packed = True if algorithm == ConvAlgorithm.Direct else False

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )

    test[DType.float32, algorithm, packed](
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
        rt,
    )


fn main():
    with Runtime() as rt:
        test_shapes[ConvAlgorithm.Im2Col](rt)
        test_shapes[ConvAlgorithm.Direct](rt)
