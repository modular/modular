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

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.resize import (
    CoordinateTransformationMode,
    _AxisTaps,
    RoundMode,
    resize_cubic,
    resize_linear,
    resize_nearest_neighbor,
)
from std.testing import assert_almost_equal, assert_equal
from std.utils import IndexList


def _pattern(n: Int) -> List[Float32]:
    var values = List[Float32](capacity=n)
    for i in range(n):
        values.append(Float32((i * 37) % 256))
    return values^


def _resize_cubic_f32(
    src: List[Float32], in_shape: IndexList[4], out_shape: IndexList[4]
) -> List[Float32]:
    var output = List[Float32](length=out_shape.flattened_length(), fill=0)
    resize_cubic(
        TileTensor(src.unsafe_ptr(), row_major(Coord(in_shape))),
        TileTensor(output.unsafe_ptr(), row_major(Coord(out_shape))),
    )
    return output^


def _cubic_weight(x: Float64) -> Float64:
    comptime a = -0.5
    var t = abs(x)
    if t < 1:
        return ((a + 2) * t - (a + 3)) * t * t + 1
    if t < 2:
        return ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return 0


def _reference_taps(in_dim: Int, out_dim: Int) -> List[Float64]:
    """Returns the [out_dim, in_dim] tap matrix of one axis, in float64."""
    var taps = List[Float64](length=out_dim * in_dim, fill=0)
    var step = Float64(in_dim) / Float64(out_dim)
    for i in range(out_dim):
        var center = (Float64(i) + 0.5) * step
        var stretch = max(step, 1.0)
        var lo = max(Int(center - 2 * stretch + 0.5), 0)
        var hi = min(Int(center + 2 * stretch + 0.5), in_dim)
        var total: Float64 = 0
        for j in range(lo, hi):
            total += _cubic_weight((Float64(j) + 0.5 - center) / stretch)
        for j in range(lo, hi):
            taps[i * in_dim + j] = (
                _cubic_weight((Float64(j) + 0.5 - center) / stretch) / total
            )
    return taps^


def _tap_window(in_dim: Int, out_dim: Int, i: Int) -> Tuple[Int, Int]:
    """Returns the first and last input index output i can read, exclusive.

    Mirrors the window of _reference_taps, so the resize loop below visits
    only the taps that can be nonzero instead of scanning the whole axis.
    """
    var step = Float64(in_dim) / Float64(out_dim)
    var center = (Float64(i) + 0.5) * step
    var stretch = max(step, 1.0)
    return (
        max(Int(center - 2 * stretch + 0.5), 0),
        min(Int(center + 2 * stretch + 0.5), in_dim),
    )


def _reference_resize(
    src: List[Float32], in_shape: IndexList[4], out_shape: IndexList[4]
) -> List[Float64]:
    """Resizes W, then H, of src with a dense tap matrix in float64."""
    var data = List[Float64](capacity=len(src))
    for v in src:
        data.append(v.cast[.float64]())
    var shape = in_shape
    for d in range(2, 0, -1):
        var n_in = shape[d]
        var n_out = out_shape[d]
        if n_in == n_out:
            continue
        var taps = _reference_taps(n_in, n_out)
        var outer = 1
        for e in range(d):
            outer *= shape[e]
        var inner = 1
        for e in range(d + 1, 4):
            inner *= shape[e]
        var next = List[Float64](length=outer * n_out * inner, fill=0)
        for o in range(outer):
            for i in range(n_out):
                var (lo, hi) = _tap_window(n_in, n_out, i)
                for j in range(lo, hi):
                    var w = taps[i * n_in + j]
                    if w == 0:
                        continue
                    for c in range(inner):
                        next[(o * n_out + i) * inner + c] += (
                            w * data[(o * n_in + j) * inner + c]
                        )
        data = next^
        shape[d] = n_out
    return data^


def _check_cubic_matches_reference(
    in_shape: IndexList[4], out_shape: IndexList[4]
) raises:
    var src = _pattern(in_shape.flattened_length())
    var got = _resize_cubic_f32(src, in_shape, out_shape)
    var want = _reference_resize(src, in_shape, out_shape)
    for i in range(len(got)):
        # A layout or tap bug is off by whole levels on this 0 to 255 data.
        assert_almost_equal(got[i].cast[.float64](), want[i], atol=1e-2)


def _check_cubic_layouts_agree(
    n: Int, h: Int, w: Int, c: Int, out_h: Int, out_w: Int
) raises:
    """Checks that channels-last, channels-first, one frame at a time, and W
    then H as two calls all give the same bits."""
    var hwc = _pattern(n * h * w * c)
    var chw = List[Float32](length=len(hwc), fill=0)
    for b in range(n):
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    chw[((b * c + ch) * h + y) * w + x] = hwc[
                        ((b * h + y) * w + x) * c + ch
                    ]
    var want = _resize_cubic_f32(
        hwc, IndexList[4](n, h, w, c), IndexList[4](n, out_h, out_w, c)
    )

    var got_chw = _resize_cubic_f32(
        chw, IndexList[4](n * c, h, w, 1), IndexList[4](n * c, out_h, out_w, 1)
    )
    for b in range(n):
        for y in range(out_h):
            for x in range(out_w):
                for ch in range(c):
                    assert_equal(
                        got_chw[((b * c + ch) * out_h + y) * out_w + x],
                        want[((b * out_h + y) * out_w + x) * c + ch],
                    )

    var frame_len = h * w * c
    var out_frame_len = out_h * out_w * c
    for b in range(n):
        var frame = List[Float32](capacity=frame_len)
        for i in range(frame_len):
            frame.append(hwc[b * frame_len + i])
        var got = _resize_cubic_f32(
            frame, IndexList[4](1, h, w, c), IndexList[4](1, out_h, out_w, c)
        )
        for i in range(out_frame_len):
            assert_equal(got[i], want[b * out_frame_len + i])

    var mid = _resize_cubic_f32(
        hwc, IndexList[4](n, h, w, c), IndexList[4](n, h, out_w, c)
    )
    var got = _resize_cubic_f32(
        mid, IndexList[4](n, h, out_w, c), IndexList[4](n, out_h, out_w, c)
    )
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def _check_cubic_uint8_matches_float32(
    in_shape: IndexList[4], out_shape: IndexList[4]
) raises:
    var as_f32 = _pattern(in_shape.flattened_length())
    var as_u8 = List[UInt8](capacity=len(as_f32))
    for v in as_f32:
        as_u8.append(v.cast[.uint8]())
    var want = _resize_cubic_f32(as_f32, in_shape, out_shape)
    var got = List[Float32](length=out_shape.flattened_length(), fill=0)
    resize_cubic(
        TileTensor(as_u8.unsafe_ptr(), row_major(Coord(in_shape))),
        TileTensor(got.unsafe_ptr(), row_major(Coord(out_shape))),
    )
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    if in_shape == out_shape:
        for i in range(len(got)):
            assert_equal(got[i], as_f32[i])


def _naive_cubic(
    src: List[Float32], in_shape: IndexList[4], out_shape: IndexList[4]
) -> List[Float32]:
    """Resizes W, then H, of src with one scalar FMA per tap in tap order.
    This is the rounding every fast path in resize_cubic must reproduce bit
    for bit."""
    var data = src.copy()
    var shape = in_shape
    for d in range(2, 0, -1):
        var n_in = shape[d]
        var n_out = out_shape[d]
        var taps = _AxisTaps(n_in, n_out)
        var outer = 1
        for e in range(d):
            outer *= shape[e]
        var inner = 1
        for e in range(d + 1, 4):
            inner *= shape[e]
        var next = List[Float32](length=outer * n_out * inner, fill=0)
        for o in range(outer):
            for i in range(n_out):
                var weights = taps.weights(i)
                for c in range(inner):
                    var acc = Float32(0)
                    for k in range(len(weights)):
                        var j = taps.start_of(i) + k
                        acc = data[(o * n_in + j) * inner + c].fma(
                            weights[k], acc
                        )
                    next[(o * n_out + i) * inner + c] = acc
        data = next^
        shape[d] = n_out
    return data^


def _check_cubic_matches_naive(
    in_shape: IndexList[4], out_shape: IndexList[4]
) raises:
    var src = _pattern(in_shape.flattened_length())
    var got = _resize_cubic_f32(src, in_shape, out_shape)
    var want = _naive_cubic(src, in_shape, out_shape)
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def main() raises:
    var ctx = DeviceContext(api="cpu")

    def test_upsample_sizes_nearest_1(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_nearest_1")
        var input_stack: Array[Float32, 4] = [Float32(1), 2, 3, 4]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

        var output_stack = Array[Float32, 24](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 4, 6]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output, ctx)

        for i in range(24):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_upsample_sizes_nearest_1
    # CHECK: 1.0,1.0,1.0,2.0,2.0,2.0,1.0,1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,4.0,4.0,3.0,3.0,3.0,4.0,4.0,4.0,
    test_upsample_sizes_nearest_1(ctx)

    def test_downsample_sizes_nearest(ctx: DeviceContext) raises:
        print("== test_downsample_sizes_nearest")
        var input_stack: Array[Float32, 8] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 4]())

        var output_stack = Array[Float32, 2](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 1, 2]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output, ctx)

        for i in range(2):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_downsample_sizes_nearest
    # CHECK: 1.0,3.0,
    test_downsample_sizes_nearest(ctx)

    def test_downsample_sizes_nearest_half_pixel_1D(
        ctx: DeviceContext,
    ) raises:
        print("== test_downsample_sizes_nearest_half_pixel_1D")
        var input_stack: Array[Float32, 16] = [
            Float32(0),
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 4, 4]())

        var output_stack = Array[Float32, 2](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 1, 2]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel1D, RoundMode.HalfDown
        ](input, output, ctx)

        for i in range(2):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_downsample_sizes_nearest_half_pixel_1D
    # CHECK: 0.0,2.0,
    test_downsample_sizes_nearest_half_pixel_1D(ctx)

    def test_upsample_sizes_nearest_2(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_nearest_2")
        var input_stack: Array[Float32, 4] = [Float32(1), 2, 3, 4]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

        var output_stack = Array[Float32, 56](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 7, 8]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel, RoundMode.HalfDown
        ](input, output, ctx)

        for i in range(56):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_upsample_sizes_nearest_2
    # CHECK: 1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,
    test_upsample_sizes_nearest_2(ctx)

    def test_upsample_sizes_nearest_floor_align_corners(
        ctx: DeviceContext,
    ) raises:
        print("== test_upsample_sizes_nearest_floor_align_corners")
        var input_stack: Array[Float32, 16] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 4, 4]())

        var output_stack = Array[Float32, 64](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 8, 8]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.AlignCorners, RoundMode.Floor
        ](input, output, ctx)

        for i in range(64):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_upsample_sizes_nearest_floor_align_corners
    # CHECK: 1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,1.0,1.0,1.0,2.0,2.0,3.0,3.0,4.0,5.0,5.0,5.0,6.0,6.0,7.0,7.0,8.0,5.0,5.0,5.0,6.0,6.0,7.0,7.0,8.0,9.0,9.0,9.0,10.0,10.0,11.0,11.0,12.0,9.0,9.0,9.0,10.0,10.0,11.0,11.0,12.0,13.0,13.0,13.0,14.0,14.0,15.0,15.0,16.0,
    test_upsample_sizes_nearest_floor_align_corners(ctx)

    def test_upsample_sizes_nearest_round_half_up_asymmetric(
        ctx: DeviceContext,
    ) raises:
        print("== test_upsample_sizes_nearest_round_half_up_asymmetric")
        var input_stack: Array[Float32, 16] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 4, 4]())

        var output_stack = Array[Float32, 64](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 8, 8]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.Asymmetric, RoundMode.HalfUp
        ](input, output, ctx)

        for i in range(64):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_upsample_sizes_nearest_round_half_up_asymmetric
    # CHECK: 1.0,2.0,2.0,3.0,3.0,4.0,4.0,4.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,
    test_upsample_sizes_nearest_round_half_up_asymmetric(ctx)

    def test_upsample_sizes_nearest_ceil_half_pixel(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_nearest_ceil_half_pixel")
        var input_stack: Array[Float32, 16] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 4, 4]())

        var output_stack = Array[Float32, 64](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 8, 8]())

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel, RoundMode.Ceil
        ](input, output, ctx)

        for i in range(64):
            print(output_stack[i], end=",")
        print("")

    # CHECK-LABEL: test_upsample_sizes_nearest_ceil_half_pixel
    # CHECK: 1.0,2.0,2.0,3.0,3.0,4.0,4.0,4.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,5.0,6.0,6.0,7.0,7.0,8.0,8.0,8.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,9.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,13.0,14.0,14.0,15.0,15.0,16.0,16.0,16.0,
    test_upsample_sizes_nearest_ceil_half_pixel(ctx)

    def test_upsample_sizes_linear(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_linear")
        var input_stack: Array[Float32, 4] = [Float32(1), 2, 3, 4]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

        var output_stack = Array[Float32, 16](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 4, 4]())

        # TORCH REFERENCE:
        # x = np.array([[[[1, 2], [3, 4]]]])
        # y = torch.nn.functional.interpolate(torch.Tensor(x), (4, 4), mode="bilinear")
        # print(y.flatten())

        var reference_stack: Array[Float32, 16] = [
            Float32(1.0000),
            1.2500,
            1.7500,
            2.0000,
            1.5000,
            1.7500,
            2.2500,
            2.5000,
            2.5000,
            2.7500,
            3.2500,
            3.5000,
            3.0000,
            3.2500,
            3.7500,
            4.0000,
        ]

        resize_linear[CoordinateTransformationMode.HalfPixel, False](
            input, output
        )

        for i in range(16):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_upsample_sizes_linear
    # CHECK-NOT: ASSERT ERROR
    test_upsample_sizes_linear(ctx)

    def test_upsample_sizes_linear_align_corners(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_linear_align_corners")
        var input_stack: Array[Float32, 4] = [Float32(1), 2, 3, 4]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

        var output_stack = Array[Float32, 16](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 4, 4]())

        # TORCH REFERENCE:
        # x = np.array([[[[1, 2], [3, 4]]]])
        # y = torch.nn.functional.interpolate(
        # torch.Tensor(x), (4, 4), mode="bilinear", align_corners=True)
        # print(y.flatten())
        var reference_stack: Array[Float32, 16] = [
            Float32(1.0000),
            1.3333,
            1.6667,
            2.0000,
            1.6667,
            2.0000,
            2.3333,
            2.6667,
            2.3333,
            2.6667,
            3.0000,
            3.3333,
            3.0000,
            3.3333,
            3.6667,
            4.0000,
        ]

        resize_linear[CoordinateTransformationMode.AlignCorners, False](
            input, output
        )

        for i in range(16):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_upsample_sizes_linear_align_corners
    # CHECK-NOT: ASSERT ERROR
    test_upsample_sizes_linear_align_corners(ctx)

    def test_downsample_sizes_linear(ctx: DeviceContext) raises:
        print("== test_downsample_sizes_linear")
        var input_stack: Array[Float32, 8] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 4]())

        var output_stack = Array[Float32, 2](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 1, 2]())
        # TORCH REFERENCE:
        # x = np.arange(1, 9).reshape((1, 1, 2, 4))
        # y = torch.nn.functional.interpolate(torch.Tensor(x), (1, 2), mode="bilinear")
        # print(y.flatten())
        var reference_stack: Array[Float32, 2] = [
            Float32(3.50000),
            5.50000,
        ]

        resize_linear[CoordinateTransformationMode.HalfPixel, False](
            input, output
        )

        for i in range(2):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_downsample_sizes_linear
    # CHECK-NOT: ASSERT ERROR
    test_downsample_sizes_linear(ctx)

    def test_downsample_sizes_linear_align_corners(ctx: DeviceContext) raises:
        print("== test_downsample_sizes_linear_align_corners")
        var input_stack: Array[Float32, 8] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 4]())

        var output_stack = Array[Float32, 2](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 1, 2]())
        # TORCH REFERENCE:
        # x = np.arange(1, 9).reshape((1, 1, 2, 4))
        # y = torch.nn.functional.interpolate(
        #     torch.Tensor(x), (1, 2), mode="bilinear", align_corners=True
        # )
        # print(y.flatten())
        var reference_stack: Array[Float32, 2] = [Float32(1), 4]

        resize_linear[CoordinateTransformationMode.AlignCorners, False](
            input, output
        )

        for i in range(2):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_downsample_sizes_linear_align_corners
    # CHECK-NOT: ASSERT ERROR
    test_downsample_sizes_linear_align_corners(ctx)

    def test_upsample_sizes_trilinear(ctx: DeviceContext) raises:
        print("== test_upsample_sizes_trilinear")
        var input_stack: Array[Float32, 16] = [
            Float32(0),
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        var input = TileTensor(input_stack, row_major[1, 4, 2, 2]())

        var output_stack = Array[Float32, 96](fill={})
        var output = TileTensor(output_stack, row_major[1, 6, 4, 4]())

        # TORCH REFERENCE:
        # x = np.arange(16).reshape((1, 1, 4, 2, 2))
        # y = torch.nn.functional.interpolate(
        #     torch.Tensor(x), (6, 4, 4), mode="trilinear"
        # )
        # print(y.flatten())
        # fmt: off
        var reference_stack: Array[Float32, 96] = [
            Float32(0.00000),  0.25000,  0.75000,  1.00000,  0.50000,  0.75000,  1.25000,
            1.50000,  1.50000,  1.75000,  2.25000,  2.50000,  2.00000,  2.25000,
            2.75000,  3.00000,  2.00000,  2.25000,  2.75000,  3.00000,  2.50000,
            2.75000,  3.25000,  3.50000,  3.50000,  3.75000,  4.25000,  4.50000,
            4.00000,  4.25000,  4.75000,  5.00000,  4.66667,  4.91667,  5.41667,
            5.66667,  5.16667,  5.41667,  5.91667,  6.16667,  6.16667,  6.41667,
            6.91667,  7.16667,  6.66667,  6.91667,  7.41667,  7.66667,  7.33333,
            7.58333,  8.08333,  8.33333,  7.83333,  8.08333,  8.58333,  8.83333,
            8.83333,  9.08333,  9.58333,  9.83333,  9.33333,  9.58333, 10.08333,
            10.33333, 10.00000, 10.25000, 10.75000, 11.00000, 10.50000, 10.75000,
            11.25000, 11.50000, 11.50000, 11.75000, 12.25000, 12.50000, 12.00000,
            12.25000, 12.75000, 13.00000, 12.00000, 12.25000, 12.75000, 13.00000,
            12.50000, 12.75000, 13.25000, 13.50000, 13.50000, 13.75000, 14.25000,
            14.50000, 14.00000, 14.25000, 14.75000, 15.00000
        ]
        # fmt: on

        resize_linear[CoordinateTransformationMode.HalfPixel, False](
            input, output
        )

        for i in range(96):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_upsample_sizes_trilinear
    # CHECK-NOT: ASSERT ERROR
    test_upsample_sizes_trilinear(ctx)

    def test_downsample_sizes_linear_antialias(ctx: DeviceContext) raises:
        print("== test_downsample_sizes_linear_antialias")
        var input_stack: Array[Float32, 16] = [
            Float32(0),
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        var input = TileTensor(input_stack, row_major[1, 1, 4, 4]())

        var output_stack = Array[Float32, 4](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 2, 2]())

        # TORCH REFERENCE:
        # x = np.arange(16).reshape((1, 1, 4, 4))
        # y = torch.nn.functional.interpolate(
        #     torch.Tensor(x), (2, 2), mode="bilinear", antialias=True
        # )
        # print(y.flatten())
        var reference_stack: Array[Float32, 4] = [
            Float32(3.57143),
            5.14286,
            9.85714,
            11.42857,
        ]

        resize_linear[CoordinateTransformationMode.HalfPixel, True](
            input, output
        )

        for i in range(4):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    # CHECK-LABEL: test_downsample_sizes_linear_antialias
    # CHECK-NOT: ASSERT ERROR
    test_downsample_sizes_linear_antialias(ctx)

    def test_no_resize(ctx: DeviceContext) raises:
        print("== test_no_resize")
        var input_stack: Array[Float32, 4] = [Float32(1), 1, 1, 1]
        var input = TileTensor(input_stack, row_major[1, 1, 2, 2]())

        var output_stack = Array[Float32, 4](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 2, 2]())

        var reference_stack: Array[Float32, 4] = [
            Float32(1.0000),
            1.0000,
            1.0000,
            1.0000,
        ]

        resize_linear[CoordinateTransformationMode.HalfPixel, False](
            input, output
        )

        for i in range(4):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-4
            )

    test_no_resize(ctx)

    def test_downsample_sizes_cubic_antialias() raises:
        print("== test_downsample_sizes_cubic_antialias")
        var input_stack = Array[Float32, 16](fill={})
        for i in range(16):
            input_stack[i] = Float32(i)
        var input = TileTensor(input_stack, row_major[1, 4, 4, 1]())

        var output_stack = Array[Float32, 4](fill={})
        var output = TileTensor(output_stack, row_major[1, 2, 2, 1]())

        # TORCH REFERENCE:
        # x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        # y = torch.nn.functional.interpolate(
        #     x, (2, 2), mode="bicubic", antialias=True
        # )
        # fmt: off
        var reference_stack: Array[Float32, 4] = [Float32(2.9338841), 4.7603307, 10.2396698, 12.0661154]
        # fmt: on

        resize_cubic(input, output)

        for i in range(4):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-5
            )

    test_downsample_sizes_cubic_antialias()

    def test_upsample_sizes_cubic_antialias() raises:
        print("== test_upsample_sizes_cubic_antialias")
        var input_stack: Array[Float32, 9] = [
            Float32(0),
            5,
            1,
            7,
            2,
            9,
            3,
            8,
            4,
        ]
        var input = TileTensor(input_stack, row_major[1, 3, 3, 1]())

        var output_stack = Array[Float32, 20](fill={})
        var output = TileTensor(output_stack, row_major[1, 5, 4, 1]())

        # TORCH REFERENCE:
        # x = torch.tensor([[0., 5, 1], [7, 2, 9], [3, 8, 4]])[None, None]
        # y = torch.nn.functional.interpolate(
        #     x, (5, 4), mode="bicubic", antialias=True
        # )
        # The edge rows and columns drop out-of-range taps and renormalize.
        # fmt: off
        var reference_stack: Array[Float32, 20] = [Float32(-0.8289212), 3.4173713, 3.8273115, 0.1439034, 2.5797508, 3.2645378, 3.8833196, 4.0481739, 7.2611942, 3.3751166, 4.2619271, 9.3656721, 4.6618390, 5.3466277, 5.9654093, 6.1302619, 2.3974934, 6.6437860, 7.0537267, 3.3703182]
        # fmt: on

        resize_cubic(input, output)

        for i in range(20):
            assert_almost_equal(
                output_stack[i], reference_stack[i], atol=1e-5, rtol=1e-5
            )

    test_upsample_sizes_cubic_antialias()

    def test_downsample_cubic_antialias_near_max() raises:
        print("== test_downsample_cubic_antialias_near_max")
        # Unnormalized taps for a 4x downscale sum to about 4, so dividing
        # by the tap sum only after accumulating would overflow here.
        var input_stack = Array[Float32, 8](fill=3e38)
        var input = TileTensor(input_stack, row_major[1, 1, 8, 1]())

        var output_stack = Array[Float32, 2](fill={})
        var output = TileTensor(output_stack, row_major[1, 1, 2, 1]())

        resize_cubic(input, output)

        for i in range(2):
            assert_almost_equal(output_stack[i], 3e38, rtol=1e-6)

    test_downsample_cubic_antialias_near_max()

    def test_cubic_matches_reference() raises:
        print("== test_cubic_matches_reference")
        # Many narrow slabs along a long axis.
        _check_cubic_matches_reference(
            IndexList[4](2, 30, 1000, 3), IndexList[4](2, 13, 999, 3)
        )
        # One element per slab.
        _check_cubic_matches_reference(
            IndexList[4](3, 40, 1000, 1), IndexList[4](3, 17, 999, 1)
        )
        # Slabs a little wider than a vector.
        _check_cubic_matches_reference(
            IndexList[4](1, 20, 50, 17), IndexList[4](1, 9, 23, 17)
        )
        # Width only, and height only.
        _check_cubic_matches_reference(
            IndexList[4](2, 30, 40, 3), IndexList[4](2, 30, 21, 3)
        )
        _check_cubic_matches_reference(
            IndexList[4](2, 30, 40, 3), IndexList[4](2, 70, 40, 3)
        )
        # Size 1 in and out.
        _check_cubic_matches_reference(
            IndexList[4](1, 1, 7, 3), IndexList[4](1, 5, 1, 3)
        )

    test_cubic_matches_reference()

    def test_cubic_layouts_agree() raises:
        print("== test_cubic_layouts_agree")
        _check_cubic_layouts_agree(n=4, h=21, w=37, c=3, out_h=9, out_w=16)
        _check_cubic_layouts_agree(n=2, h=13, w=11, c=3, out_h=40, out_w=29)
        _check_cubic_layouts_agree(n=2, h=21, w=37, c=16, out_h=9, out_w=16)

    test_cubic_layouts_agree()

    def test_cubic_uint8_matches_float32() raises:
        print("== test_cubic_uint8_matches_float32")
        # H and W together, W alone, H alone, and no resize at all.
        _check_cubic_uint8_matches_float32(
            IndexList[4](2, 21, 37, 3), IndexList[4](2, 9, 16, 3)
        )
        _check_cubic_uint8_matches_float32(
            IndexList[4](1, 21, 37, 16), IndexList[4](1, 9, 16, 16)
        )
        _check_cubic_uint8_matches_float32(
            IndexList[4](1, 21, 37, 3), IndexList[4](1, 21, 16, 3)
        )
        _check_cubic_uint8_matches_float32(
            IndexList[4](1, 21, 37, 3), IndexList[4](1, 9, 37, 3)
        )
        _check_cubic_uint8_matches_float32(
            IndexList[4](1, 5, 7, 3), IndexList[4](1, 5, 7, 3)
        )

    test_cubic_uint8_matches_float32()

    def test_cubic_fast_paths_match_naive() raises:
        print("== test_cubic_fast_paths_match_naive")
        # Each channel count takes a different unrolled transposed pass, the
        # one channel at a time fallback, or the slab pass once it fills a
        # vector.
        var channels: List[Int] = [1, 2, 3, 4, 7, 16, 17]
        for c in channels:
            _check_cubic_matches_naive(
                IndexList[4](2, 37, 53, c), IndexList[4](2, 19, 70, c)
            )
        # A width pass over many narrow slabs, with no height pass.
        _check_cubic_matches_naive(
            IndexList[4](1, 20, 9, 7), IndexList[4](1, 20, 5, 7)
        )
        # Channels first, so the width pass has one element per slab.
        _check_cubic_matches_naive(
            IndexList[4](3, 40, 61, 1), IndexList[4](3, 17, 29, 1)
        )
        # A long axis over enough slabs for the transposed pass, so it needs
        # many windows.
        _check_cubic_matches_naive(
            IndexList[4](1, 16, 3840, 3), IndexList[4](1, 16, 448, 3)
        )
        # Fewer columns than a vector, so they move one at a time.
        _check_cubic_matches_naive(
            IndexList[4](1, 20, 7, 1), IndexList[4](1, 20, 5, 1)
        )
        # 387 columns end 3 past a multiple of the ring size, for 8- and
        # 16-lane vectors, so the last vector would wrap the ring.
        _check_cubic_matches_naive(
            IndexList[4](1, 16, 129, 3), IndexList[4](1, 16, 100, 3)
        )

    test_cubic_fast_paths_match_naive()
