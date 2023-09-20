# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from math import round_half_down, round_half_up, floor, ceil, min
from algorithm.functional import elementwise
from runtime.llcl import OutputChainPtr


@value
struct CoordinateTransformationMode:
    var value: Int
    alias HalfPixel = CoordinateTransformationMode(0)
    alias AlignCorners = CoordinateTransformationMode(1)
    alias Asymmetric = CoordinateTransformationMode(2)

    @always_inline
    fn __eq__(self, other: CoordinateTransformationMode) -> Bool:
        return self.value == other.value


@value
struct RoundMode:
    var value: Int
    alias HalfDown = RoundMode(0)
    alias HalfUp = RoundMode(1)
    alias Floor = RoundMode(2)
    alias Ceil = RoundMode(3)

    @always_inline
    fn __eq__(self, other: RoundMode) -> Bool:
        return self.value == other.value


fn resize_nearest_neighbor[
    coordinate_transformation_mode: CoordinateTransformationMode,
    round_mode: RoundMode,
    rank: Int,
    type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    var scales = StaticTuple[rank, Float32]()
    for i in range(rank):
        scales[i] = (output.dim(i) / input.dim(i)).cast[DType.float32]()

    @parameter
    @always_inline
    fn coord_transform(out_coord: Int, dim: Int) -> Float32:
        @parameter
        if (
            coordinate_transformation_mode
            == CoordinateTransformationMode.HalfPixel
        ):
            # note: coordinates are for the CENTER of the pixel
            # - 0.5 term at the end is so that when we round to the nearest integer
            # coordinate, we get the coordinate whose center is closest
            return (out_coord + 0.5) / scales[dim] - 0.5
        elif (
            coordinate_transformation_mode
            == CoordinateTransformationMode.AlignCorners
        ):
            # note: resized image will have same corners as original image
            return (
                out_coord
                * ((input.dim(dim) - 1) / (output.dim(dim) - 1)).cast[
                    DType.float32
                ]()
            )
        elif (
            coordinate_transformation_mode
            == CoordinateTransformationMode.Asymmetric
        ):
            return out_coord / scales[dim]
        else:
            constrained[
                True, "coordinate_transformation_mode not implemented"
            ]()
            return 0

    @parameter
    @always_inline
    fn round[type: DType](val: SIMD[type, 1]) -> SIMD[type, 1]:
        @parameter
        if round_mode == RoundMode.HalfDown:
            return round_half_down(val)
        elif round_mode == RoundMode.HalfUp:
            return round_half_up(val)
        elif round_mode == RoundMode.Floor:
            return floor(val)
        elif round_mode == RoundMode.Ceil:
            return ceil(val)
        else:
            constrained[True, "round_mode not implemented"]()
            return val

    @parameter
    fn nn_interpolate[
        simd_width: Int, _rank: Int
    ](out_coords: StaticIntTuple[_rank]):
        var in_coords = StaticIntTuple[rank](0)

        @unroll
        for i in range(rank):
            in_coords[i] = min(
                round(coord_transform(out_coords[i], i)).to_int(),
                input.dim(i) - 1,
            )

        output[rebind[StaticIntTuple[rank]](out_coords)] = input[in_coords]

    # TODO (#21439): can use memcpy when scale on inner dimension is 1
    elementwise[rank, 1, nn_interpolate](output.get_shape(), out_chain)
