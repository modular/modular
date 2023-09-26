# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from math import round_half_down, round_half_up, floor, ceil, min, max
from algorithm.functional import elementwise
from algorithm.reduction import _get_nd_indices_from_flat_index
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


@parameter
@always_inline
fn coord_transform[
    mode: CoordinateTransformationMode
](out_coord: Int, in_dim: Int, out_dim: Int, scale: Float32) -> Float32:
    @parameter
    if mode == CoordinateTransformationMode.HalfPixel:
        # note: coordinates are for the CENTER of the pixel
        # - 0.5 term at the end is so that when we round to the nearest integer
        # coordinate, we get the coordinate whose center is closest
        return (out_coord + 0.5) / scale - 0.5
    elif mode == CoordinateTransformationMode.AlignCorners:
        # aligning "corners" when output is 1D isn't well defined
        # this matches pytorch
        if out_dim == 1:
            return 0
        # note: resized image will have same corners as original image
        return out_coord * ((in_dim - 1) / (out_dim - 1)).cast[DType.float32]()
    elif mode == CoordinateTransformationMode.Asymmetric:
        return out_coord / scale
    else:
        constrained[True, "coordinate_transformation_mode not implemented"]()
        return 0


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
                round(
                    coord_transform[coordinate_transformation_mode](
                        out_coords[i], input.dim(i), output.dim(i), scales[i]
                    )
                ).to_int(),
                input.dim(i) - 1,
            )

        output[rebind[StaticIntTuple[rank]](out_coords)] = input[in_coords]

    # TODO (#21439): can use memcpy when scale on inner dimension is 1
    elementwise[rank, 1, nn_interpolate](output.get_shape(), out_chain)


@always_inline
fn linear_filter(x: Float32) -> Float32:
    """This is a tent filter.

    f(x) = 1 + x, x < 0
    f(x) = 1 - x, 0 <= x < 1
    f(x) = 0, x >= 1

    """
    var coeff = x
    if x < 0:
        coeff = -x
    if x < 1:
        return 1 - coeff
    return 0


@parameter
@always_inline
fn interpolate_point_1d[
    coordinate_transformation_mode: CoordinateTransformationMode,
    rank: Int,
    type: DType,
](
    dim: Int,
    out_coords: StaticIntTuple[rank],
    scale: Float32,
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    let center = coord_transform[coordinate_transformation_mode](
        out_coords[dim], input.dim(dim), output.dim(dim), scale
    ) + 0.5
    let xmin = max(0, floor(center - 1 + 0.5).to_int())
    let xmax = min(input.dim(dim), floor(center + 1 + 0.5).to_int())
    var in_coords = out_coords
    var sum = SIMD[type, 1](0)
    var acc = SIMD[type, 1](0)
    for k in range(0, xmax - xmin):
        in_coords[dim] = k + xmin
        let dist_from_center = k + xmin - center + 0.5
        let filter_coeff = linear_filter(dist_from_center).cast[type]()
        acc += input[in_coords] * filter_coeff
        sum += filter_coeff

    # normalize to handle cases near image boundary where only 1 point is used
    # for interpolation
    for k in range(0, xmax - xmin):
        acc /= sum

    output[out_coords] = acc


fn resize_linear[
    coordinate_transformation_mode: CoordinateTransformationMode,
    rank: Int,
    type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    """Resizes input to output shape using linear interpolation.

    Does not use anti-aliasing filter for downsampling (coming soon).

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        rank: Rank of the input and output.
        type: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
        out_chain: The chain to attach our results to.


    """
    _resize_with_filter[linear_filter, coordinate_transformation_mode](
        input, output, out_chain
    )


fn _resize_with_filter[
    filter: fn (Float32) -> Float32,
    coordinate_transformation_mode: CoordinateTransformationMode,
    rank: Int,
    type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    var scales = StaticTuple[rank, Float32]()

    var resize_dims = InlinedFixedVector[rank, Int](rank)
    for i in range(rank):
        if input.dim(i) != output.dim(i):
            resize_dims.append(i)
        scales[i] = (output.dim(i) / input.dim(i)).cast[DType.float32]()

    var in_ptr = input.data
    var out_ptr = DTypePointer[type]()
    var out_in_use = False
    var tmp_buffer = DTypePointer[type]()
    # ping pong between using out_buffer and tmp_buffer to store outputs
    # of 1d interpolation pass accross one of the dimensions
    if len(resize_dims) == 1:
        out_ptr = output.data  # avoid allocating tmp_buffer
    # last iteration must use output
    elif len(resize_dims) % 2 == 0:
        tmp_buffer = DTypePointer[type].alloc(output.num_elements())
        out_ptr = tmp_buffer
        out_in_use = False
    else:
        tmp_buffer = DTypePointer[type].alloc(output.num_elements())
        out_ptr = output.data
        out_in_use = True
    var in_shape = input.get_shape()
    var out_shape = input.get_shape()
    # interpolation is separable, so perform 1d interpolation across each
    # interpolated dimension
    for dim_idx in range(len(resize_dims)):
        let resize_dim = resize_dims[dim_idx]
        out_shape[resize_dim] = output.dim(resize_dim)

        let in_buf = NDBuffer[rank, DimList.create_unknown[rank](), type](
            in_ptr, in_shape
        )
        let out_buf = NDBuffer[rank, DimList.create_unknown[rank](), type](
            out_ptr, out_shape
        )

        let num_rows = out_buf.num_elements() // out_shape[resize_dim]
        for row_idx in range(num_rows):
            var coords = _get_nd_indices_from_flat_index(
                row_idx, out_shape, resize_dim
            )
            for i in range(out_shape[resize_dim]):
                coords[resize_dim] = i
                interpolate_point_1d[coordinate_transformation_mode](
                    resize_dim, coords, scales[resize_dim], in_buf, out_buf
                )

        in_shape = out_shape
        in_ptr = out_ptr

        out_ptr = tmp_buffer if out_in_use else output.data
        out_in_use = not out_in_use

    tmp_buffer.free()
    resize_dims._del_old()

    out_chain.mark_ready()
