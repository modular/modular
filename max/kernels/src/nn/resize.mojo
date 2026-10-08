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
from std.bit import bit_width
from std.math import align_down, align_up, ceil, ceildiv, floor


from max.algorithm.functional import elementwise
from max.algorithm.reduction import _get_nd_indices_from_flat_index
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from std.memory import unsafe_memcpy
from std.sys import simd_width_of, size_of

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
    var outer = in_shape[0]
    var inner = in_shape[3]
    _resize_2d_fused(
        _AxisTaps(in_shape[2], out_shape[2]),
        _AxisTaps(in_shape[1], out_shape[1]),
        input.reshape(row_major((outer, in_shape[1], in_shape[2] * inner))),
        output.reshape(row_major((outer, out_shape[1], out_shape[2] * inner))),
        in_shape[2],
        out_shape[2],
        inner,
    )


comptime _WIDTH = simd_width_of[DType.float32]()
"""Float32 lanes in a vector, the width every cubic pass works in."""

comptime _TILE_BYTES = 16384
"""Bytes of input window a pass aims to keep resident in L1."""

comptime _PIPE_COLS = 64
"""Input columns a transposed pass moves between bursts of taps."""

comptime _GROUP = 8
"""Independent accumulators a pass interleaves to cover FMA latency."""

comptime _MAX_UNROLLED_CHANNELS = 4
"""Most vectors per input position a transposed pass unrolls."""

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
    var max_count: Int

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
        self.max_count = 0
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
        self.max_count = max(self.max_count, self.count_of(len(self.start) - 1))

    @inline(.always)
    def start_of(self, i: Int) -> Int:
        """Returns the first input that output i reads."""
        return self.start.unsafe_get(i)

    @inline(.always)
    def count_of(self, i: Int) -> Int:
        """Returns how many inputs output i reads."""
        return self.offset.unsafe_get(i + 1) - self.offset.unsafe_get(i)

    @inline(.always)
    def end_of(self, i: Int) -> Int:
        """Returns the input after the last one output i reads."""
        return self.start_of(i) + self.count_of(i)

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
def _weighted_row_sum[
    wraps: Bool
](
    taps: _AxisTaps,
    i: Int,
    rows: TileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
    begin: Int,
    end: Int,
):
    """Writes columns [begin, end) of row i of dst as a weighted sum of input
    rows. begin is a multiple of _GROUP vectors. With wraps, rows is a ring
    of a power-of-two size holding input row r at r % size."""
    comptime block = _GROUP * _WIDTH
    debug_assert(begin % block == 0, "begin must be block aligned")
    var first = taps.start_of(i)
    var n = taps.count_of(i)
    var weights = taps.weights(i)
    var mask = Int(rows.dim[0]()) - 1
    var c = begin
    # Halving the group for what is left keeps several accumulators in
    # flight when a row is narrower than _GROUP vectors, which happens at
    # fewer elements the wider the vectors are.
    comptime for halvings in range(bit_width(_GROUP)):
        comptime group = _GROUP >> halvings
        comptime span = group * _WIDTH
        while c + span <= end:
            # A view per group keeps each vector's offset a compile-time
            # constant, so the loads need no per-vector index registers.
            var cols = rows.tile(
                Coord(rows.dim[0](), Idx[span]), (0, c // span)
            )
            var acc = Array[SIMD[DType.float32, _WIDTH], group](fill=0)
            for k in range(n):
                var w = SIMD[DType.float32, _WIDTH](weights.unsafe_get(k))
                var r = (first + k) & mask if wraps else first + k
                comptime for q in range(group):
                    acc[q] = cols.load[width=_WIDTH]((r, q * _WIDTH)).fma(
                        w, acc[q]
                    )
            comptime for q in range(group):
                dst.store((i, c + q * _WIDTH), acc[q])
            c += span
    while c < end:
        var acc = Float32(0)
        for k in range(n):
            var r = (first + k) & mask if wraps else first + k
            acc = rows.load[width=1]((r, c)).fma(weights.unsafe_get(k), acc)
        dst.store((i, c), acc)
        c += 1


def _lane_group_mask[
    width: Int, *, halves: Bool, high: Bool
]() -> IndexList[width]:
    """Returns a two-vector shuffle that works within each group of 4 lanes.

    It interleaves the low or high pairs of the two vectors. With halves, it
    joins their low or high halves instead.
    """
    var mask = IndexList[width]()
    for g in range(0, width, 4):
        var lo = g + (2 if high else 0)
        comptime if halves:
            mask[g], mask[g + 1] = lo, lo + 1
            mask[g + 2], mask[g + 3] = width + lo, width + lo + 1
        else:
            mask[g], mask[g + 1] = lo, width + lo
            mask[g + 2], mask[g + 3] = lo + 1, width + lo + 1
    return mask


@inline(.always)
def _load_lane_groups[
    dtype: DType, //, width: Int
](src: TileTensor[dtype, ...], row: Int, col: Int) -> SIMD[dtype, width]:
    """Loads columns [col, col + 4) of rows row, row + 4, ... into
    consecutive groups of 4 lanes."""
    comptime if width == 4:
        return rebind[SIMD[dtype, width]](src.load[width=4]((row, col)))
    else:
        comptime half = width // 2
        return rebind[SIMD[dtype, width]](
            _load_lane_groups[half](src, row, col).join(
                _load_lane_groups[half](src, row + half, col)
            )
        )


@inline(.always)
def _load_transposed[
    dtype: DType, //, width: Int
](src: TileTensor[dtype, ...], row: Int, col: Int) -> Array[
    SIMD[dtype, width], width
]:
    """Loads the width x width block of src at (row, col) as one vector per
    column.

    This is a transpose of 4 x 4 blocks. Moving whole blocks folds into
    4-wide loads, which leaves shuffles that stay within groups of 4 lanes.
    Every vector ISA does those in one cheap instruction.
    """
    comptime assert width % 4 == 0, "width must be a multiple of 4"
    comptime pairs_lo = _lane_group_mask[width, halves=False, high=False]()
    comptime pairs_hi = _lane_group_mask[width, halves=False, high=True]()
    comptime halves_lo = _lane_group_mask[width, halves=True, high=False]()
    comptime halves_hi = _lane_group_mask[width, halves=True, high=True]()
    var cols = Array[SIMD[dtype, width], width](fill=0)
    comptime for c in range(0, width, 4):
        var r0 = _load_lane_groups[width](src, row, col + c)
        var r1 = _load_lane_groups[width](src, row + 1, col + c)
        var r2 = _load_lane_groups[width](src, row + 2, col + c)
        var r3 = _load_lane_groups[width](src, row + 3, col + c)
        var t0 = r0.shuffle[pairs_lo](r1)
        var t1 = r0.shuffle[pairs_hi](r1)
        var t2 = r2.shuffle[pairs_lo](r3)
        var t3 = r2.shuffle[pairs_hi](r3)
        cols[c] = t0.shuffle[halves_lo](t2)
        cols[c + 1] = t0.shuffle[halves_hi](t2)
        cols[c + 2] = t1.shuffle[halves_lo](t3)
        cols[c + 3] = t1.shuffle[halves_hi](t3)
    return cols^


@inline(.always)
def _move_block[
    to_tile: Bool
](
    src: TileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
    col: Int,
    slot: Int,
):
    """Moves _WIDTH columns, starting at column col and tile row slot, in the
    direction _move_columns describes."""
    comptime if to_tile:
        var m = _load_transposed[_WIDTH](src, 0, col)
        comptime for q in range(_WIDTH):
            dst.store((slot + q, 0), m[q])
    else:
        var m = _load_transposed[_WIDTH](src, slot, 0)
        comptime for q in range(_WIDTH):
            dst.store((q, col), m[q])


@inline(.always)
def _move_columns[
    to_tile: Bool
](
    src: TileTensor[DType.float32, ...],
    dst: MutTileTensor[DType.float32, ...],
    begin: Int,
    end: Int,
    partial: Bool,
) -> Int:
    """Moves columns [begin, end) of _WIDTH rows into a tile, or back out of
    the tile when not to_tile, and returns the end of the columns moved.

    The tile holds one column per row. It is a ring: column j is in tile row
    j % ring, where ring is the tile's row count. begin and ring are
    multiples of _WIDTH, and columns before begin must already be moved.
    Without partial, only whole vectors of columns move.
    """
    var ring: Int
    comptime if to_tile:
        ring = Int(dst.dim[0]())
    else:
        ring = Int(src.dim[0]())
    var j = begin
    var slot = begin % ring
    while j + _WIDTH <= end:
        _move_block[to_tile](src, dst, j, slot)
        j += _WIDTH
        slot += _WIDTH
        if slot == ring:
            slot = 0
    if partial:
        while j < end:
            comptime for s in range(_WIDTH):
                comptime if to_tile:
                    dst.store((j % ring, s), src.load[width=1]((s, j)))
                else:
                    dst.store((s, j), src.load[width=1]((j % ring, s)))
            j += 1
    return j


@inline(.always)
def _add_tap[
    n: Int, //, channels: Int, first: Int
](
    mut acc: Array[SIMD[DType.float32, _WIDTH], n],
    taps: _AxisTaps,
    tile: TileTensor[DType.float32, ...],
    mask: Int,
    i: Int,
    k: Int,
):
    """Adds tap k of output i to accumulators [first, first + channels)."""
    var w = SIMD[DType.float32, _WIDTH](taps.weights(i).unsafe_get(k))
    var p = (taps.start_of(i) + k) & mask
    comptime for ch in range(channels):
        acc[first + ch] = tile.load[width=_WIDTH]((p, ch * _WIDTH)).fma(
            w, acc[first + ch]
        )


def _tap_pixels(channels: Int) -> Int:
    """Returns how many outputs a transposed pass interleaves, which keeps 4
    to 8 accumulators in flight."""
    if channels <= 2:
        return 4
    if channels <= _MAX_UNROLLED_CHANNELS:
        return 2
    # The pass goes one channel at a time past the counts it unrolls.
    return 1


@inline(.always)
def _apply_taps_interleaved[
    channels: Int, pixels: Int
](
    taps: _AxisTaps,
    tile: TileTensor[DType.float32, ...],
    mask: Int,
    out_tile: MutTileTensor[DType.float32, ...],
    begin: Int,
    end: Int,
    tile_first_output: Int,
):
    """Applies the taps of outputs [begin, end), pixels outputs at a time to
    keep that many independent accumulators in flight."""
    var i = begin
    while i < end:
        var group = pixels if i + pixels <= end else 1
        var acc = Array[SIMD[DType.float32, _WIDTH], pixels * channels](fill=0)
        if group == pixels:
            var shared = taps.count_of(i)
            comptime for p in range(1, pixels):
                shared = min(shared, taps.count_of(i + p))
            for k in range(shared):
                comptime for p in range(pixels):
                    _add_tap[channels, p * channels](
                        acc, taps, tile, mask, i + p, k
                    )
            comptime for p in range(pixels):
                for k in range(shared, taps.count_of(i + p)):
                    _add_tap[channels, p * channels](
                        acc, taps, tile, mask, i + p, k
                    )
        else:
            for k in range(taps.count_of(i)):
                _add_tap[channels, 0](acc, taps, tile, mask, i, k)
        comptime for p in range(pixels):
            if p < group:
                comptime for ch in range(channels):
                    out_tile.store(
                        (i + p - tile_first_output, ch * _WIDTH),
                        acc[p * channels + ch],
                    )
        i += group


@inline(.always)
def _apply_taps_to_tile(
    taps: _AxisTaps,
    tile: TileTensor[DType.float32, ...],
    out_tile: MutTileTensor[DType.float32, ...],
    begin: Int,
    end: Int,
    tile_first_output: Int,
):
    """Applies the taps of outputs [begin, end) to a ring of a power-of-two
    number of input positions, one per row, writing output i to row
    i - tile_first_output of out_tile."""
    var mask = Int(tile.dim[0]()) - 1
    var channels = Int(tile.dim[1]()) // _WIDTH
    comptime for c in range(1, _MAX_UNROLLED_CHANNELS + 1):
        if channels == c:
            _apply_taps_interleaved[c, _tap_pixels(c)](
                taps,
                tile.reshape(row_major((Int(tile.dim[0]()), Idx[c * _WIDTH]))),
                mask,
                out_tile.reshape(
                    row_major((Int(out_tile.dim[0]()), Idx[c * _WIDTH]))
                ),
                begin,
                end,
                tile_first_output,
            )
            return
    for ch in range(channels):
        _apply_taps_interleaved[1, 1](
            taps,
            tile[:, ch * _WIDTH : (ch + 1) * _WIDTH],
            mask,
            out_tile[:, ch * _WIDTH : (ch + 1) * _WIDTH],
            begin,
            end,
            tile_first_output,
        )


@fieldwise_init
struct _Window(TrivialRegisterPassable):
    """Outputs [first_output, end_output), run as steps [first_step,
    end_step)."""

    var first_output: Int
    var end_output: Int
    var first_step: Int
    var end_step: Int


@fieldwise_init
struct _Step(TrivialRegisterPassable):
    """Input columns [cols_begin, cols_end) to transpose, and the output the
    step finishes up to, outputs_end."""

    var cols_begin: Int
    var cols_end: Int
    var outputs_end: Int


struct _TransposedPlan(Movable):
    """The schedule of a transposed pass.

    A transposed pass resamples one slab per vector lane. It moves the
    slabs, transposed, into a ring of ring_positions input positions. Each
    position is inner rows of the ring, one vector each, that share a tap
    weight. Input columns move into the ring once each, in order, skipping
    only those no output reads.

    Outputs are cut into windows whose results fit in _TILE_BYTES. A window
    alternates transposing a few columns with finishing the outputs they
    complete, so memory traffic overlaps independent FMAs.
    """

    var ring_positions: Int
    var windows: List[_Window]
    var steps: List[_Step]
    var inner: Int
    var lookahead: Int
    var position_bytes: Int

    def __init__(
        out self,
        taps: _AxisTaps,
        in_dim: Int,
        out_dim: Int,
        inner: Int,
        slabs: Int,
    ):
        """Plans the pass, or nothing when the slab pass runs instead:
        an inner that fills a vector, a width that keeps its size, or
        fewer slabs than lanes never reaches the transposed schedule."""
        self.ring_positions = 0
        self.windows = List[_Window]()
        self.steps = List[_Step]()
        self.inner = inner
        # Columns move in whole vectors, so the ring holds up to a vector's
        # worth past the last position a window reads.
        self.lookahead = ceildiv(_WIDTH, inner)
        self.position_bytes = inner * _WIDTH * size_of[DType.float32]()
        if inner >= _WIDTH or in_dim == out_dim or slabs < _WIDTH:
            return
        # A power of two makes a position a mask, and a multiple of the
        # vector width keeps a vector of columns from wrapping.
        self.ring_positions = _WIDTH
        while (
            self.ring_positions < taps.max_count + self.lookahead
            or 2 * self.ring_positions * self.position_bytes <= _TILE_BYTES
        ):
            self.ring_positions *= 2
        var transposed = 0
        var first = 0
        while first < len(taps.start):
            var end = self._window_end(taps, first)
            transposed = self._add_window(
                taps, first, end, in_dim * inner, transposed
            )
            first = end

    def _window_end(self, taps: _AxisTaps, first: Int) -> Int:
        """Returns the end of the window that starts at output first: as many
        outputs as the ring and _TILE_BYTES hold, and at least one."""
        var lo = taps.start_of(first)
        var hi = taps.end_of(first)
        var end = first + 1
        while end < len(taps.start):
            var next_hi = max(hi, taps.end_of(end))
            if (
                next_hi - lo + self.lookahead > self.ring_positions
                or (end + 1 - first) * self.position_bytes > _TILE_BYTES
            ):
                break
            hi = next_hi
            end += 1
        return end

    def _add_window(
        mut self,
        taps: _AxisTaps,
        first: Int,
        end: Int,
        total: Int,
        var transposed: Int,
    ) -> Int:
        """Adds the steps of the window of outputs [first, end), given that
        the first transposed of the total input columns have moved. Returns
        how many have moved after it."""
        var lo = taps.start_of(first)
        var hi = 0
        for i in range(first, end):
            hi = max(hi, taps.end_of(i))
        var pixels = _tap_pixels(self.inner)
        transposed = max(transposed, align_down(lo * self.inner, _WIDTH))
        var needed = min(total, align_up(hi * self.inner, _WIDTH))
        var first_step = len(self.steps)
        var finished = first
        while finished < end:
            var cols_begin = transposed
            var cols_end = max(transposed, min(needed, transposed + _PIPE_COLS))
            # Leftover columns move only at the end of the input.
            if cols_end == total:
                transposed = total
            else:
                transposed += align_down(cols_end - transposed, _WIDTH)
            var ready = transposed // self.inner
            var step_first = finished
            while finished < end and taps.end_of(finished) <= ready:
                finished += 1
            # Whole groups only until the window's last outputs, so outputs
            # keep their accumulators interleaved.
            if finished < end:
                finished = step_first + align_down(
                    finished - step_first, pixels
                )
            self.steps.append(_Step(cols_begin, cols_end, finished))
        self.windows.append(_Window(first, end, first_step, len(self.steps)))
        return transposed


struct _SlabPass(Movable):
    """Resamples one dimension of independent [in_dim, inner] slabs."""

    var taps: _AxisTaps
    var in_dim: Int
    var out_dim: Int
    var inner: Int
    var strip: Int
    var plan: _TransposedPlan
    var tile: List[Float32]
    var out_tile: List[Float32]

    def __init__(
        out self,
        var taps: _AxisTaps,
        in_dim: Int,
        out_dim: Int,
        inner: Int,
        slabs: Int,
    ):
        comptime block = _GROUP * _WIDTH
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inner = inner
        # Column strips of a wide slab whose input window stays in L1.
        self.strip = max(
            block,
            align_down(
                _TILE_BYTES
                // (max(taps.max_count, 1) * size_of[DType.float32]()),
                block,
            ),
        )
        self.plan = _TransposedPlan(taps, in_dim, out_dim, inner, slabs)
        self.tile = List[Float32](
            unsafe_uninit_length=self.plan.ring_positions * inner * _WIDTH
        )
        self.out_tile = List[Float32](
            # A plan's window always takes one output, which may overflow
            # _TILE_BYTES.
            unsafe_uninit_length=(
                max(
                    _TILE_BYTES // size_of[DType.float32](), inner * _WIDTH
                ) if self.plan.ring_positions
                > 0 else 0
            )
        )
        self.taps = taps^

    def batch(self, slabs_left: Int) -> Int:
        """Returns how many slabs the next resize takes: a vector of them on
        the transposed path, while that many are left."""
        return _WIDTH if self.inner < _WIDTH and slabs_left >= _WIDTH else 1

    @inline(.always)
    def resize(
        mut self,
        src: TileTensor[DType.float32, ...],
        dst: MutTileTensor[DType.float32, ...],
    ):
        """Resamples each row of src, a flattened slab, into the same row of
        dst. Takes as many rows as batch returned. One row runs the slab
        path, and a vector of rows runs the transposed path."""
        if Int(src.dim[0]()) != 1:
            return self._resize_transposed(src, dst)
        var rows = src[Idx[0], :].reshape(row_major((self.in_dim, self.inner)))
        var out_rows = dst[Idx[0], :].reshape(
            row_major((self.out_dim, self.inner))
        )
        var col = 0
        while col < self.inner:
            var col_end = min(self.inner, col + self.strip)
            for i in range(self.out_dim):
                _weighted_row_sum[wraps=False](
                    self.taps, i, rows, out_rows, col, col_end
                )
            col = col_end

    def _resize_transposed(
        mut self,
        src: TileTensor[DType.float32, ...],
        dst: MutTileTensor[DType.float32, ...],
    ):
        """Resamples the rows of src, a vector of flattened slabs, into those
        of dst, one slab per vector lane, for inner too narrow to fill a
        vector."""
        var steps = Span(self.plan.steps)
        var position_len = self.inner * _WIDTH
        var ring_positions = self.plan.ring_positions
        var tile_cols = TileTensor(
            self.tile, row_major((ring_positions * self.inner, Idx[_WIDTH]))
        )
        var tile_positions = TileTensor(
            self.tile, row_major((ring_positions, position_len))
        )
        var out_cols = TileTensor(
            self.out_tile,
            row_major((len(self.out_tile) // _WIDTH, Idx[_WIDTH])),
        )
        var out_positions = TileTensor(
            self.out_tile,
            row_major((len(self.out_tile) // position_len, position_len)),
        )
        var total = Int(src.dim[1]())
        for window in self.plan.windows:
            var lo = window.first_output * self.inner
            var window_dst = dst[:, lo : window.end_output * self.inner]
            var finished = window.first_output
            var stored = 0
            for s in range(window.first_step, window.end_step):
                var step = steps.unsafe_get(s)
                _ = _move_columns[to_tile=True](
                    src,
                    tile_cols,
                    step.cols_begin,
                    step.cols_end,
                    partial=step.cols_end == total,
                )
                if step.outputs_end > finished:
                    _apply_taps_to_tile(
                        self.taps,
                        tile_positions,
                        out_positions,
                        finished,
                        step.outputs_end,
                        window.first_output,
                    )
                    finished = step.outputs_end
                stored = _move_columns[to_tile=False](
                    out_cols,
                    window_dst,
                    stored,
                    (finished - window.first_output) * self.inner,
                    partial=False,
                )
            _ = _move_columns[to_tile=False](
                out_cols,
                window_dst,
                stored,
                (window.end_output - window.first_output) * self.inner,
                partial=True,
            )


def _cast_rows[
    in_dtype: DType, dtype: DType, //
](src: TileTensor[in_dtype, ...], dst: MutTileTensor[dtype, ...]):
    """Copies [rows, cols] src into dst, converting each element."""
    for r in range(Int(src.dim[0]())):

        def cast_at[width: Int](c: Int) {imm}:
            dst.store((r, c), src.load[width=width]((r, c)).cast[dtype]())

        vectorize[simd_width_of[dtype]()](Int(src.dim[1]()), cast_at)


def _resize_rows[
    in_dtype: DType, //
](
    mut w_pass: _SlabPass,
    src: TileTensor[in_dtype, ...],
    dst: MutTileTensor[DType.float32, ...],
    stage: MutTileTensor[DType.float32, ...],
):
    """Resamples W of a batch of input rows into dst. Integer rows convert
    into stage first. An unchanged W converts or copies the rows."""
    if w_pass.in_dim == w_pass.out_dim:
        _cast_rows(src, dst)
        return
    comptime if in_dtype == DType.float32:
        w_pass.resize(src.bitcast[DType.float32](), dst)
    else:
        var rows = Int(src.dim[0]())
        _cast_rows(src, stage[0:rows, :])
        w_pass.resize(stage[0:rows, :], dst)


def _resize_2d_fused[
    in_dtype: DType, //
](
    var taps_w: _AxisTaps,
    taps_h: _AxisTaps,
    src: TileTensor[in_dtype, ...],
    dst: MutTileTensor[DType.float32, ...],
    in_w: Int,
    out_w: Int,
    inner: Int,
):
    """Resamples W, then H, of [outer, H, W * inner] views of row-major
    [outer, H, W, inner] tensors, keeping only the W-resampled rows the H
    filter still needs in a ring. Integer input is converted a batch of rows
    at a time, as the W pass reaches it. An unchanged H writes the W pass
    straight into dst.
    """
    var mid_row = out_w * inner
    var in_h = Int(src.dim[1]())
    var same_h = in_h == Int(dst.dim[1]())
    var w_pass = _SlabPass(taps_w^, in_w, out_w, inner, in_h)
    # Filling batch rows while the previous max_count - 1 are live needs
    # max_count + batch - 1 slots. A power of two makes a slot a mask, and a
    # multiple of batch keeps an aligned batch from wrapping.
    var batch = w_pass.batch(in_h)
    var capacity = batch
    while capacity < taps_h.max_count + batch - 1:
        capacity *= 2
    var ring_buffer = List[Float32](
        unsafe_uninit_length=0 if same_h else capacity * mid_row
    )
    var ring = TileTensor(ring_buffer, row_major((capacity, mid_row)))
    var in_row = Int(src.dim[2]())
    var stage_buffer = List[Float32](
        unsafe_uninit_length=0 if in_dtype == DType.float32 else batch * in_row
    )
    var stage = TileTensor(stage_buffer, row_major((batch, in_row)))
    for o in range(Int(src.dim[0]())):
        var src_rows = src[o, :, :]
        var dst_rows = dst[o, :, :]
        var produced = 0
        for i in range(Int(dst.dim[1]())):
            while produced < taps_h.end_of(i):
                var rows = w_pass.batch(in_h - produced)
                var batch_rows = src_rows[produced : produced + rows, :]
                if same_h:
                    _resize_rows(
                        w_pass,
                        batch_rows,
                        dst_rows[produced : produced + rows, :],
                        stage,
                    )
                else:
                    var slot = produced & (capacity - 1)
                    _resize_rows(
                        w_pass, batch_rows, ring[slot : slot + rows, :], stage
                    )
                produced += rows
            if not same_h:
                _weighted_row_sum[wraps=True](
                    taps_h, i, ring, dst_rows, 0, mid_row
                )
