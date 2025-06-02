# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import List
from math import ceil, floor

from algorithm.functional import elementwise
from algorithm.reduction import _get_nd_indices_from_flat_index
from buffer import NDBuffer
from memory import UnsafePointer, memcpy

from utils import IndexList, StaticTuple


@value
struct CoordinateTransformationMode:
    var value: Int
    alias HalfPixel = CoordinateTransformationMode(0)
    alias AlignCorners = CoordinateTransformationMode(1)
    alias Asymmetric = CoordinateTransformationMode(2)
    alias HalfPixel1D = CoordinateTransformationMode(3)

    @always_inline
    @implicit
    fn __init__(out self, value: Int):
        self.value = value

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
        return (out_coord + Float32(0.5)) / scale - 0.5
    elif mode == CoordinateTransformationMode.HalfPixel1D:
        # Same as HalfPixel except for 1D output. Described here:
        # https://onnx.ai/onnx/operators/onnx__Resize.html
        if out_dim == 1:
            return 0
        return (out_coord + Float32(0.5)) / scale - 0.5
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
    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    @always_inline
    fn __eq__(self, other: RoundMode) -> Bool:
        return self.value == other.value


@value
struct InterpolationMode:
    var value: Int
    alias Linear = InterpolationMode(0)

    @always_inline
    fn __eq__(self, other: InterpolationMode) -> Bool:
        return self.value == other.value


@value
@register_passable("trivial")
struct Interpolator[mode: InterpolationMode]:
    var cubic_coeff: Float32

    @always_inline
    @implicit
    fn __init__(out self, cubic_coeff: Float32):
        self.cubic_coeff = cubic_coeff

    @always_inline
    fn __init__(out self):
        self.cubic_coeff = 0

    @staticmethod
    @always_inline
    fn filter_length() -> Int:
        @parameter
        if mode == InterpolationMode.Linear:
            return 1
        else:
            constrained[False, "InterpolationMode not supported"]()
            return -1

    @always_inline
    fn filter(self, x: Float32) -> Float32:
        @parameter
        if mode == InterpolationMode.Linear:
            return linear_filter(x)
        else:
            constrained[False, "InterpolationMode not supported"]()
            return -1


fn resize_nearest_neighbor[
    coordinate_transformation_mode: CoordinateTransformationMode,
    round_mode: RoundMode,
    rank: Int,
    type: DType,
](input: NDBuffer[type, rank], output: NDBuffer[mut=True, type, rank]) raises:
    var scales = StaticTuple[Float32, rank]()
    for i in range(rank):
        scales[i] = (output.dim(i) / input.dim(i)).cast[DType.float32]()

    @parameter
    @always_inline
    fn round[type: DType](val: Scalar[type]) -> Scalar[type]:
        @parameter
        if round_mode == RoundMode.HalfDown:
            return ceil(val - 0.5)
        elif round_mode == RoundMode.HalfUp:
            return floor(val + 0.5)
        elif round_mode == RoundMode.Floor:
            return floor(val)
        elif round_mode == RoundMode.Ceil:
            return ceil(val)
        else:
            constrained[True, "round_mode not implemented"]()
            return val

    @__copy_capture(scales)
    @parameter
    fn nn_interpolate[
        simd_width: Int, _rank: Int
    ](out_coords: IndexList[_rank]):
        var in_coords = IndexList[rank](0)

        @parameter
        for i in range(rank):
            in_coords[i] = min(
                Int(
                    round(
                        coord_transform[coordinate_transformation_mode](
                            out_coords[i],
                            input.dim(i),
                            output.dim(i),
                            scales[i],
                        )
                    )
                ),
                input.dim(i) - 1,
            )

        output[rebind[IndexList[rank]](out_coords)] = input[in_coords]

    # TODO (#21439): can use memcpy when scale on inner dimension is 1
    elementwise[nn_interpolate, 1](output.get_shape())


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
    antialias: Bool,
    rank: Int,
    type: DType,
    interpolation_mode: InterpolationMode,
](
    interpolator: Interpolator[interpolation_mode],
    dim: Int,
    out_coords: IndexList[rank],
    scale: Float32,
    input: NDBuffer[type, rank],
    output: NDBuffer[mut=True, type, rank],
):
    var center = (
        coord_transform[coordinate_transformation_mode](
            out_coords[dim], input.dim(dim), output.dim(dim), scale
        )
        + 0.5
    )
    var filter_scale = 1 / scale if antialias and scale < 1 else 1
    var support = interpolator.filter_length() * filter_scale
    var xmin = max(0, Int(center - support + 0.5))
    var xmax = min(input.dim(dim), Int(center + support + 0.5))
    var in_coords = out_coords
    var sum = Scalar[type](0)
    var acc = Scalar[type](0)
    var ss = 1 / filter_scale
    for k in range(xmax - xmin):
        in_coords[dim] = k + xmin
        var dist_from_center = ((k + xmin + Float32(0.5)) - center) * ss
        var filter_coeff = interpolator.filter(dist_from_center).cast[type]()
        acc += input[in_coords] * filter_coeff
        sum += filter_coeff

    # normalize to handle cases near image boundary where only 1 point is used
    # for interpolation
    output[out_coords] = acc / sum


fn resize_linear[
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    rank: Int,
    type: DType,
](input: NDBuffer[type, rank], output: NDBuffer[mut=True, type, rank]):
    """Resizes input to output shape using linear interpolation.

    Does not use anti-aliasing filter for downsampling (coming soon).

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        antialias: Whether or not to use an antialiasing linear/cubic filter, which when downsampling, uses
            more points to avoid aliasing artifacts. Effectively stretches the filter by a factor of 1 / scale.
        rank: Rank of the input and output.
        type: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.


    """
    _resize[
        InterpolationMode.Linear, coordinate_transformation_mode, antialias
    ](input, output)


fn _resize[
    interpolation_mode: InterpolationMode,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    rank: Int,
    type: DType,
](input: NDBuffer[type, rank], output: NDBuffer[mut=True, type, rank]):
    if input.get_shape() == output.get_shape():
        return memcpy(output.data, input.data, input.size())
    var scales = StaticTuple[Float32, rank]()
    var resize_dims = List[Int, hint_trivial_type=True](capacity=rank)
    var tmp_dims = IndexList[rank](0)
    for i in range(rank):
        # need to consider output dims when upsampling and input dims when downsampling
        tmp_dims[i] = max(input.dim(i), output.dim(i))
        scales[i] = (output.dim(i) / input.dim(i)).cast[DType.float32]()
        if input.dim(i) != output.dim(i):
            resize_dims.append(i)
    var interpolator = Interpolator[interpolation_mode]()

    var in_ptr = input.data
    var out_ptr = UnsafePointer[Scalar[type]]()
    var using_tmp1 = False
    var tmp_buffer1 = UnsafePointer[Scalar[type]]()
    var tmp_buffer2 = UnsafePointer[Scalar[type]]()
    # ping pong between using tmp_buffer1 and tmp_buffer2 to store outputs
    # of 1d interpolation pass across one of the dimensions
    if len(resize_dims) == 1:  # avoid allocating tmp_buffer
        out_ptr = output.data
    if len(resize_dims) > 1:  # avoid allocating second tmp_buffer
        tmp_buffer1 = UnsafePointer[Scalar[type]].alloc(
            tmp_dims.flattened_length()
        )
        out_ptr = tmp_buffer1
        using_tmp1 = True
    if len(resize_dims) > 2:  # need a second tmp_buffer
        # TODO: if you are upsampling all dims, you can use the output in place of tmp_buffer2
        # as long as you make sure that the last iteration uses tmp1_buffer as the input
        # and tmp_buffer2 (output) as the output
        tmp_buffer2 = UnsafePointer[Scalar[type]].alloc(
            tmp_dims.flattened_length()
        )
    var in_shape = input.get_shape()
    var out_shape = input.get_shape()
    # interpolation is separable, so perform 1d interpolation across each
    # interpolated dimension
    for dim_idx in range(len(resize_dims)):
        if dim_idx == len(resize_dims) - 1:
            out_ptr = output.data
        var resize_dim = resize_dims[dim_idx]
        out_shape[resize_dim] = output.dim(resize_dim)

        var in_buf = NDBuffer[type, rank](in_ptr, in_shape)
        var out_buf = NDBuffer[type, rank](out_ptr, out_shape)

        var num_rows = out_buf.num_elements() // out_shape[resize_dim]
        for row_idx in range(num_rows):
            var coords = _get_nd_indices_from_flat_index(
                row_idx, out_shape, resize_dim
            )
            for i in range(out_shape[resize_dim]):
                coords[resize_dim] = i
                interpolate_point_1d[coordinate_transformation_mode, antialias](
                    interpolator,
                    resize_dim,
                    coords,
                    scales[resize_dim],
                    in_buf,
                    out_buf,
                )

        in_shape = out_shape
        in_ptr = out_ptr

        out_ptr = tmp_buffer2 if using_tmp1 else tmp_buffer1
        using_tmp1 = not using_tmp1

    if tmp_buffer1:
        tmp_buffer1.free()
    if tmp_buffer2:
        tmp_buffer2.free()
