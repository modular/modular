# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from tensor import Tensor, TensorShape
from test_utils import linear_fill
from Resize import (
    resize_nearest_neighbor,
    CoordinateTransformationMode,
    RoundMode,
)
from runtime.llcl import Runtime, OwningOutputChainPtr


fn test_case[
    rank: Int,
    coord_transform: CoordinateTransformationMode,
    round_mode: RoundMode,
    type: DType,
](
    input: Tensor[type],
    output: Tensor[type],
    scales: StaticTuple[rank, Float32],
):
    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        resize_nearest_neighbor[coord_transform, round_mode](
            input._to_ndbuffer[rank](),
            output._to_ndbuffer[rank](),
            scales,
            out_chain.borrow(),
        )
        out_chain.wait()

    for i in range(output.num_elements()):
        print_no_newline(output._to_buffer()[i])
        print_no_newline(",")
    print("")


fn test_case[
    rank: Int,
    coord_transform: CoordinateTransformationMode,
    round_mode: RoundMode,
    type: DType,
](input: Tensor[type], output: Tensor[type]):
    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        resize_nearest_neighbor[coord_transform, round_mode,](
            input._to_ndbuffer[rank](),
            output._to_ndbuffer[rank](),
            out_chain.borrow(),
        )
        out_chain.wait()

    for i in range(output.num_elements()):
        print_no_newline(output._to_buffer()[i])
        print_no_newline(",")
    print("")


fn main():
    fn test_upsample_scales_nearest():
        print("== test_upsample_scales_nearest")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 2, 2)
        linear_fill[type](input, VariadicList[SIMD[type, 1]](1, 2, 3, 4))
        let scales = StaticTuple[4, Float32](1, 1, 2, 3)
        let output = Tensor[type](1, 1, 4, 6)
        test_case[
            4, CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output, scales)

    # CHECK-LABEL: test_upsample_scales_nearest
    # CHECK: 1.0,1.0,1.0,2.0,2.0,2.0,1.0,1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,4.0,4.0,3.0,3.0,3.0,4.0,4.0,4.0,
    test_upsample_scales_nearest()

    fn test_downsample_scales_nearest():
        print("== test_downsample_scales_nearest")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 2, 4)
        linear_fill[type](
            input, VariadicList[SIMD[type, 1]](1, 2, 3, 4, 5, 6, 7, 8)
        )
        let output = Tensor[type](1, 1, 1, 2)
        let scales = StaticTuple[4, Float32](1, 1, 0.6, 0.6)

        test_case[
            4, CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output, scales)

    # CHECK-LABEL: test_downsample_scales_nearest
    # CHECK: 1.0,3.0,
    test_downsample_scales_nearest()

    fn test_upsample_sizes_nearest():
        print("== test_upsample_sizes_nearest")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 2, 2)
        linear_fill[type](input, VariadicList[SIMD[type, 1]](1, 2, 3, 4))
        let output = Tensor[type](1, 1, 7, 8)

        test_case[
            4, CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output)

    # CHECK-LABEL: test_upsample_sizes_nearest
    # CHECK: 1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,
    test_upsample_sizes_nearest()

    fn test_upsample_sizes_nearest_floor_align_corners():
        print("== test_upsample_sizes_nearest_floor_align_corners")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 4, 4)
        linear_fill[type](
            input,
            VariadicList[SIMD[type, 1]](
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ),
        )
        let output = Tensor[type](1, 1, 8, 8)

        test_case[
            4, CoordinateTransformationMode.AlignCorners, RoundMode.Floor
        ](input, output)

    # CHECK-LABEL: test_upsample_sizes_nearest_floor_align_corners
    # CHECK: 1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,5.0,5.0,5.0,6.0,6.0,7.0,7.0,8.0,5.0,5.0,5.0,6.0,6.0,7.0,7.0,8.0,9.0,9.0,9.0,10.0,10.0,11.0,11.0,12.0,9.0,9.0,9.0,10.0,10.0,11.0,11.0,12.0,13.0,13.0,13.0,14.0,14.0,15.0,15.0,16.0,
    test_upsample_sizes_nearest_floor_align_corners()

    fn test_upsample_sizes_nearest_round_half_up_asymmetric():
        print("== test_upsample_sizes_nearest_round_half_up_asymmetric")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 4, 4)
        linear_fill[type](
            input,
            VariadicList[SIMD[type, 1]](
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ),
        )
        let output = Tensor[type](1, 1, 8, 8)

        test_case[4, CoordinateTransformationMode.Asymmetric, RoundMode.HalfUp](
            input, output
        )

    # CHECK-LABEL: test_upsample_sizes_nearest_round_half_up_asymmetric
    # CHECK: 1.0,2.0,2.0,3.0,3.0,4.0,4.0,4.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,
    test_upsample_sizes_nearest_round_half_up_asymmetric()

    fn test_upsample_sizes_nearest_ceil_half_pixel():
        print("== test_upsample_sizes_nearest_ceil_half_pixel")
        alias type = DType.float32
        let input = Tensor[type](1, 1, 4, 4)
        linear_fill[type](
            input,
            VariadicList[SIMD[type, 1]](
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ),
        )
        let output = Tensor[type](1, 1, 8, 8)

        test_case[4, CoordinateTransformationMode.HalfPixel, RoundMode.Ceil](
            input, output
        )

    # CHECK-LABEL: test_upsample_sizes_nearest_ceil_half_pixel
    # CHECK: 1.0,2.0,2.0,3.0,3.0,4.0,4.0,4.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,
    test_upsample_sizes_nearest_ceil_half_pixel()
