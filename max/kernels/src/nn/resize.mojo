# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements tensor resize (upsample/downsample) with nearest, linear, and cubic interpolation."""

from std.algorithm import vectorize
from std.math import ceil, floor


from max.algorithm.functional import elementwise
from max.algorithm.reduction import _get_nd_indices_from_flat_index
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from std.memory import unsafe_memcpy
from std.sys import simd_width_of

from std.utils import IndexList, StaticTuple


struct CoordinateTransformationMode(ImplicitlyCopyable):
    """Specifies how output coordinates map to input coordinates during resize.
    """

    var value: Int
    comptime HalfPixel = CoordinateTransformationMode(0)
    comptime AlignCorners = CoordinateTransformationMode(1)
    comptime Asymmetric = CoordinateTransformationMode(2)
    comptime HalfPixel1D = CoordinateTransformationMode(3)

    @inline(.always)
    def __init__(out self, value: Int):
        self.value = value

    @inline(.always)
    def __eq__(self, other: CoordinateTransformationMode) -> Bool:
        return self.value == other.value


@inline(.always)
def coord_transform[
    mode: CoordinateTransformationMode
](out_coord: Int, in_dim: Int, out_dim: Int, scale: Float32) -> Float32:
    """Maps an output coordinate to an input coordinate according to the given transformation mode.

    Parameters:
        mode: The coordinate transformation mode governing the mapping.

    Args:
        out_coord: The output coordinate to map.
        in_dim: The size of the input dimension.
        out_dim: The size of the output dimension.
        scale: The ratio of output dimension size to input dimension size.

    Returns:
        The corresponding input coordinate as a floating-point value.
    """
    var out_coord_f32 = Float32(out_coord)

    comptime __match mode:
        case .HalfPixel:
            # note: coordinates are for the CENTER of the pixel
            # - 0.5 term at the end is so that when we round to the nearest integer
            # coordinate, we get the coordinate whose center is closest
            return (out_coord_f32 + Float32(0.5)) / scale - 0.5
        case .HalfPixel1D:
            # Same as HalfPixel except for 1D output. Described here:
            # https://onnx.ai/onnx/operators/onnx__Resize.html
            if out_dim == 1:
                return 0
            return (out_coord_f32 + Float32(0.5)) / scale - 0.5
        case .AlignCorners:
            # aligning "corners" when output is 1D isn't well defined
            # this matches pytorch
            if out_dim == 1:
                return 0
            # note: resized image will have same corners as original image
            return (
                out_coord_f32
                * (Float64(in_dim - 1) / Float64(out_dim - 1)).cast[.float32]()
            )
        case .Asymmetric:
            return out_coord_f32 / scale
        case _:
            comptime assert (
                False
            ), "coordinate_transformation_mode not implemented"


struct RoundMode(ImplicitlyCopyable):
    """Specifies how fractional coordinates are rounded to integer indices during nearest-neighbor resize.
    """

    var value: Int
    comptime HalfDown = RoundMode(0)
    comptime HalfUp = RoundMode(1)
    comptime Floor = RoundMode(2)
    comptime Ceil = RoundMode(3)

    @inline(.always)
    def __init__(out self, value: Int):
        self.value = value

    @inline(.always)
    def __eq__(self, other: RoundMode) -> Bool:
        return self.value == other.value


def resize_nearest_neighbor[
    coordinate_transformation_mode: CoordinateTransformationMode,
    round_mode: RoundMode,
    dtype: DType,
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Resizes input to output shape using nearest-neighbor interpolation.

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        round_mode: How to round fractional input coordinates to integer indices.
        dtype: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
        ctx: The device context used to launch the kernel.
    """
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"
    var scales = StaticTuple[Float32, input.rank]()
    for i in range(input.rank):
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()

    @inline(.always)
    def round[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
        comptime __match round_mode:
            case .HalfDown:
                return ceil(val - 0.5)
            case .HalfUp:
                return floor(val + 0.5)
            case .Floor:
                return floor(val)
            case .Ceil:
                return ceil(val)
            case _:
                comptime assert False, "round_mode not implemented"

    def nn_interpolate[
        simd_width: Int, alignment: Int = 1
    ](out_coords: Coord) {var}:
        var in_coords = IndexList[input.rank](0)

        comptime for i in range(input.rank):
            in_coords[i] = min(
                Int(
                    round(
                        coord_transform[coordinate_transformation_mode](
                            Int(out_coords[i].value()),
                            Int(input.dim(i)),
                            Int(output.dim(i)),
                            scales[i],
                        )
                    )
                ),
                Int(input.dim(i)) - 1,
            )

        var in_idx = input.layout(Coord(in_coords))
        var out_idx = output.layout(out_coords)

        output.raw_store(out_idx, input.ptr[in_idx])

    # TODO (#21439): can use unsafe_memcpy when scale on inner dimension is 1
    elementwise[1](nn_interpolate, output.layout.shape_coord(), ctx)


@inline(.always)
def linear_filter(x: Float32) -> Float32:
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


comptime _CUBIC_COEFF: Float32 = -0.5
"""The Keys cubic coefficient of `resize_cubic`, which torch and PIL use for
the antialiased filter."""


@inline(.always)
def cubic_filter(x: Float32, a: Float32) -> Float32:
    """Evaluates the Keys cubic convolution filter.

    A more negative `a` gives a sharper result.

    Args:
        x: The signed distance from the filter center.
        a: The cubic coefficient.

    Returns:
        The filter weight, zero for `|x| >= 2`.
    """
    var t = abs(x)
    if t < 1:
        return ((a + 2) * t - (a + 3)) * t * t + 1
    if t < 2:
        return ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return 0


@inline(.always)
def interpolate_point_1d[
    InputLayoutType: TensorLayout,
    //,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    dtype: DType,
](
    dim: Int,
    out_coords: IndexList[InputLayoutType.rank],
    scale: Float32,
    input: TileTensor[
        mut=False, dtype, InputLayoutType, address_space=.GENERIC, ...
    ],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
):
    """Computes one-dimensional interpolation for a single output point along a given dimension.

    Parameters:
        InputLayoutType: The layout type of the input tensor.
        coordinate_transformation_mode: The coordinate transformation mode to apply.
        antialias: Whether to stretch the filter to antialias when downsampling.
        dtype: The element type of the input and output tensors.

    Args:
        dim: The dimension along which to interpolate.
        out_coords: The multi-dimensional coordinates of the output point.
        scale: The ratio of output dimension size to input dimension size.
        input: The input tensor to read from.
        output: The output tensor to write the interpolated value to.
    """
    var center = (
        coord_transform[coordinate_transformation_mode](
            out_coords[dim], Int(input.dim(dim)), Int(output.dim(dim)), scale
        )
        + 0.5
    )
    # The tent filter has a half-width of 1, stretched by filter_scale.
    var filter_scale = 1 / scale if antialias and scale < 1 else 1
    var xmin = max(Int(center - filter_scale + 0.5), 0)
    var xmax = min(Int(input.dim(dim)), Int(center + filter_scale + 0.5))
    var in_coords = out_coords
    var sum = Scalar[dtype](0)
    var acc = Scalar[dtype](0)
    var ss = 1 / filter_scale
    for k in range(xmax - xmin):
        in_coords[dim] = k + xmin
        var dist_from_center = (
            (Float32(k + xmin) + Float32(0.5)) - center
        ) * ss
        var filter_coeff = linear_filter(dist_from_center).cast[dtype]()
        var in_idx = input.layout(Coord(in_coords))
        acc += input.raw_load(in_idx) * filter_coeff
        sum += filter_coeff

    # normalize to handle cases near image boundary where only 1 point is used
    # for interpolation
    var out_idx = output.layout(Coord(out_coords))
    output.raw_store(out_idx, acc / sum)


def resize_linear[
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    dtype: DType,
](
    input: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
):
    """Resizes input to output shape using linear interpolation.

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        antialias: Whether to stretch the linear filter by 1 / scale when
            downsampling, which reads more input points to avoid aliasing.
        dtype: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
    """
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"

    if rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    ) == rebind[IndexList[input.rank]](
        coord_to_index_list(output.layout.shape_coord())
    ):
        return unsafe_memcpy(
            dest=output.ptr, src=input.ptr, count=input.num_elements()
        )
    var scales = StaticTuple[Float32, input.rank]()
    var resize_dims = List[Int](capacity=input.rank)
    var tmp_dims = IndexList[input.rank](0)
    for i in range(input.rank):
        # need to consider output dims when upsampling and input dims when downsampling
        tmp_dims[i] = max(Int(input.dim(i)), Int(output.dim(i)))
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()
        if Int(input.dim(i)) != Int(output.dim(i)):
            resize_dims.append(i)

    var in_ptr = input.ptr.unsafe_origin_cast[MutUntrackedOrigin]()
    # SAFETY: Placeholder; always overwritten below.
    var out_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()

    var using_tmp1 = False
    var tmp_buffer1 = List[Scalar[dtype]]()
    var tmp_buffer2 = List[Scalar[dtype]]()

    # ping pong between using tmp_buffer1 and tmp_buffer2 to store outputs
    # of 1d interpolation pass across one of the dimensions
    if len(resize_dims) == 1:  # avoid allocating tmp_buffer
        out_ptr = output.ptr.unsafe_origin_cast[MutAnyOrigin]()
    if len(resize_dims) > 1:  # avoid allocating second tmp_buffer
        tmp_buffer1 = List[Scalar[dtype]](
            unsafe_uninit_length=tmp_dims.flattened_length()
        )
        out_ptr = tmp_buffer1.unsafe_ptr().as_unsafe_any_origin()
        using_tmp1 = True
    if len(resize_dims) > 2:  # need a second tmp_buffer
        # TODO: if you are upsampling all dims, you can use the output in place of tmp_buffer2
        # as long as you make sure that the last iteration uses tmp1_buffer as the input
        # and tmp_buffer2 (output) as the output
        tmp_buffer2 = List[Scalar[dtype]](
            unsafe_uninit_length=tmp_dims.flattened_length()
        )
    var in_shape = coord_to_index_list(input.layout.shape_coord())
    var out_shape = coord_to_index_list(input.layout.shape_coord())
    # interpolation is separable, so perform 1d interpolation across each
    # interpolated dimension
    for dim_idx in range(len(resize_dims)):
        if dim_idx == len(resize_dims) - 1:
            out_ptr = output.ptr.unsafe_origin_cast[MutAnyOrigin]()
        var resize_dim = resize_dims[dim_idx]
        out_shape[resize_dim] = Int(output.dim(resize_dim))

        var in_buf = TileTensor(in_ptr, row_major(Coord(in_shape)))
        var out_buf = TileTensor(out_ptr, row_major(Coord(out_shape)))

        var num_rows = out_buf.num_elements() // out_shape[resize_dim]
        for row_idx in range(num_rows):
            var coords = _get_nd_indices_from_flat_index(
                row_idx, out_shape, resize_dim
            )
            for i in range(out_shape[resize_dim]):
                coords[resize_dim] = i
                interpolate_point_1d[
                    InputLayoutType=in_buf.LayoutType,
                    coordinate_transformation_mode,
                    antialias,
                ](
                    resize_dim,
                    rebind[IndexList[in_buf.rank]](coords),
                    scales[resize_dim],
                    in_buf,
                    out_buf,
                )

        in_shape = out_shape
        in_ptr = out_ptr.unsafe_origin_cast[MutUntrackedOrigin]()

        out_ptr = (
            tmp_buffer2.unsafe_ptr() if using_tmp1 else tmp_buffer1.unsafe_ptr()
        ).as_unsafe_any_origin()
        using_tmp1 = not using_tmp1

    _ = tmp_buffer1^
    _ = tmp_buffer2^


def resize_cubic[
    in_dtype: DType, //
](
    input: ImmTileTensor[in_dtype, ...],
    output: MutTileTensor[DType.float32, ...],
):
    """Resizes the height and width of an `[outer, H, W, inner]` tensor with
    antialiased cubic interpolation.

    The filter is the Keys cubic with `a = -0.5`, stretched by the downscale
    factor. Taps outside the input are dropped, and the remaining weights are
    normalized to a sum of one. This is the filter of torch's CPU
    `interpolate(mode="bicubic", antialias=True, align_corners=False)`, and
    the result matches it to float32 rounding.

    An NHWC batch is `[N, H, W, C]`, and an NCHW batch is `[N * C, H, W, 1]`.
    W is resampled before H, and each output adds up its taps in order with
    FMAs, so every shape gives the same bits for the same pixels. The resize
    runs on the CPU, on the calling thread, in float32.

    Parameters:
        in_dtype: The type of the input: float32, or an integer type such as
            uint8. The result is the same as when you convert the input to
            float32 first.

    Args:
        input: The input to be resized, contiguous and row-major.
        output: The output containing the resized input, contiguous and
            row-major, with the outer and inner sizes of the input. It must
            not overlap the input.

    Constraints:
        Both tensors are rank 4. The input is float32 or an integer type.
    """
    comptime assert (
        input.rank == 4 and output.rank == 4
    ), "input and output must be [outer, H, W, inner]"
    comptime assert (
        in_dtype == DType.float32 or in_dtype.is_integral()
    ), "input must be float32 or an integer type"

    var in_shape = rebind[IndexList[4]](
        coord_to_index_list(input.layout.shape_coord())
    )
    var out_shape = rebind[IndexList[4]](
        coord_to_index_list(output.layout.shape_coord())
    )
    debug_assert(
        in_shape[0] == out_shape[0] and in_shape[3] == out_shape[3],
        "outer and inner sizes must match",
    )
    if input.num_elements() == 0 or output.num_elements() == 0:
        debug_assert(
            output.num_elements() == 0,
            "cannot resize an empty input to a non-empty output",
        )
        return

    comptime if in_dtype == DType.float32:
        _resize_passes(
            input.bitcast[DType.float32](), output, in_shape, out_shape
        )
    else:
        var converted = List[Float32](unsafe_uninit_length=input.num_elements())
        var flat = row_major((1, input.num_elements()))
        _cast_rows(input.reshape(flat), TileTensor(converted, flat))
        _resize_passes(
            TileTensor(converted, input.layout).as_imm(),
            output,
            in_shape,
            out_shape,
        )


comptime _CUBIC_SUPPORT = 2
"""Half-width of the cubic filter, in input pixels before any stretch."""


struct _AxisTaps(Movable):
    """Holds the filter taps of each output index along one resized dimension.

    Output i is the weighted sum of count_of(i) inputs from start[i] on, with
    weights coeffs[offset[i] : offset[i + 1]].
    """

    var start: List[Int]
    var offset: List[Int]
    var coeffs: List[Float32]

    def __init__(out self, in_dim: Int, out_dim: Int):
        """Builds the taps of a resize of in_dim inputs to out_dim: the
        normalized weights of a cubic window around each output's center,
        stretched by the downscale factor. Equal sizes give each output one
        tap of weight one, so a pass copies that axis exactly.

        The subtraction of the center is in float32, and the half-pixel shift
        and the stretch are in float64. This mix follows torch's CPU kernel,
        so the window edges and weights match it.
        """
        self.start = List[Int](capacity=out_dim)
        self.offset = List[Int](capacity=out_dim + 1)
        self.offset.append(0)
        self.coeffs = List[Float32]()
        if in_dim == out_dim:
            for i in range(out_dim):
                self.coeffs.append(1)
                self._add_output(i)
            return
        var step = Float32(in_dim) / Float32(out_dim)
        var downscaling = step >= 1
        var support = _CUBIC_SUPPORT * (step if downscaling else 1)
        var inv_stretch = Float32(1 / Float64(step)) if downscaling else 1
        self.coeffs = List[Float32](
            capacity=out_dim * (2 * Int(ceil(support)) + 1)
        )
        for i in range(out_dim):
            var center = _half_pixel_center(i, in_dim, out_dim)
            var first = max(_window_edge(center - support), 0)
            var n = max(min(in_dim, _window_edge(center + support)) - first, 0)
            var offset = len(self.coeffs)
            var sum = Float32(0)
            for k in range(n):
                var coeff = cubic_filter(
                    _tap_offset(k + first, center, inv_stretch), _CUBIC_COEFF
                )
                self.coeffs.append(coeff)
                sum += coeff
            # Normalized weights keep the accumulator in range. The raw
            # weights of a downscale by s add up to about s.
            if sum != 0:
                for k in range(offset, offset + n):
                    self.coeffs[k] /= sum
            self._add_output(first)

    def _add_output(mut self, start: Int):
        """Ends the next output's taps at the end of coeffs."""
        self.start.append(start)
        self.offset.append(len(self.coeffs))

    @inline(.always)
    def start_of(self, i: Int) -> Int:
        """Returns the first input that output i reads."""
        return self.start.unsafe_get(i)

    @inline(.always)
    def count_of(self, i: Int) -> Int:
        """Returns how many inputs output i reads."""
        return self.offset.unsafe_get(i + 1) - self.offset.unsafe_get(i)

    @inline(.always)
    def weights(self, i: Int) -> Span[Float32, origin_of(self.coeffs)]:
        """Returns the weights of output i, one per input it reads."""
        return Span(self.coeffs).unsafe_subspan(
            offset=self.offset.unsafe_get(i), length=self.count_of(i)
        )


@inline(.always)
def _half_pixel_center(i: Int, in_dim: Int, out_dim: Int) -> Float32:
    """Returns the center of output i in input pixels.

    It multiplies by the input-to-output ratio in float32, as torch does.
    coord_transform divides by the inverse ratio, which rounds differently.
    """
    return Float32(in_dim) / Float32(out_dim) * (Float32(i) + 0.5)


@inline(.always)
def _window_edge(x: Float32) -> Int:
    """Rounds a window edge, center - support or center + support, to the
    input index it starts or ends at. The shift is in float64."""
    return Int(Float64(x) + 0.5)


@inline(.always)
def _tap_offset(pos: Int, center: Float32, inv_stretch: Float32) -> Float32:
    """Returns the signed distance from center to input pos, in filter units.
    The subtraction is in float32, and the shift and the stretch are in
    float64."""
    return (
        ((Float32(pos) - center).cast[.float64]() + 0.5) * Float64(inv_stretch)
    ).cast[.float32]()


@inline(.always)
def _weighted_row_sum(
    taps: _AxisTaps,
    i: Int,
    rows: TileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
):
    """Writes row i of dst as a weighted sum of input rows."""
    var first = taps.start_of(i)
    var weights = taps.weights(i)

    def row_sum[width: Int](c: Int) {imm}:
        var acc = SIMD[DType.float32, width](0)
        for k in range(len(weights)):
            acc = rows.load[width=width]((first + k, c)).fma(
                SIMD[DType.float32, width](weights.unsafe_get(k)), acc
            )
        dst.store((i, c), acc)

    vectorize[simd_width_of[DType.float32]()](Int(rows.dim[1]()), row_sum)


def _cast_rows[
    in_dtype: DType, dtype: DType, //
](src: TileTensor[in_dtype, ...], dst: MutTileTensor[dtype, ...]):
    """Copies [rows, cols] src into dst, converting each element."""
    for r in range(Int(src.dim[0]())):

        def cast_at[width: Int](c: Int) {imm}:
            dst.store((r, c), src.load[width=width]((r, c)).cast[dtype]())

        vectorize[simd_width_of[dtype]()](Int(src.dim[1]()), cast_at)


def _resize_slabs(
    taps: _AxisTaps,
    src: TileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
    in_dim: Int,
    out_dim: Int,
    inner: Int,
):
    """Resamples each row of src, a flattened [in_dim, inner] slab, into the
    same row of dst."""
    for o in range(Int(src.dim[0]())):
        var rows = src[o, :].reshape(row_major((in_dim, inner)))
        var out_rows = dst[o, :].reshape(row_major((out_dim, inner)))
        for i in range(out_dim):
            _weighted_row_sum(taps, i, rows, out_rows)


def _resize_passes(
    src: ImmTileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
    in_shape: IndexList[4],
    out_shape: IndexList[4],
):
    """Resamples W, then H, through a temporary that holds the W-resampled
    input."""
    var outer = in_shape[0]
    var in_h = in_shape[1]
    var in_w = in_shape[2]
    var inner = in_shape[3]
    var out_h = out_shape[1]
    var out_w = out_shape[2]
    var mid = List[Float32](unsafe_uninit_length=outer * in_h * out_w * inner)
    _resize_slabs(
        _AxisTaps(in_w, out_w),
        src.reshape(row_major((outer * in_h, in_w * inner))),
        TileTensor(mid, row_major((outer * in_h, out_w * inner))),
        in_w,
        out_w,
        inner,
    )
    _resize_slabs(
        _AxisTaps(in_h, out_h),
        TileTensor(mid, row_major((outer, in_h * out_w * inner))).as_imm(),
        dst.reshape(row_major((outer, out_h * out_w * inner))),
        in_h,
        out_h,
        out_w * inner,
    )
