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

# `scalb` on the GPU against host libm, plus the subnormal inputs of the
# stdlib `test_subnormal_inputs` test, which must survive a target that
# flushes subnormal operands (Apple GPUs in float32). Results are bit-exact,
# including subnormal results on Apple GPUs.

from std.math import align_up, inf, isnan, scalb
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import TestSuite, assert_true
from test_utils.math import PairMode, libm_scalb, pair_inputs, scalb_specials

from std.utils.coord import Coord

comptime length = 8192

comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))


def _check_exact[
    dtype: DType
](
    got: Scalar[dtype],
    expected: Scalar[dtype],
    x: Scalar[dtype],
    y: Scalar[dtype],
    side: StaticString,
    i: Int,
) raises:
    # scalb is exact; comparing bits also distinguishes +0 from -0.
    var ok: Bool
    if isnan(expected):
        ok = isnan(got)
    else:
        ok = got.to_bits() == expected.to_bits()
    if not ok:
        raise Error(
            t"{side} scalb {dtype} at {i}: ({x}, {y}) got {got} expected",
            t" {expected}",
        )


def run_scalb[
    dtype: DType
](
    ctx: DeviceContext,
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    var out_device = ctx.enqueue_create_buffer[dtype](length)
    var x_buffer = Span(unsafe_ptr=x_device.unsafe_ptr(), length=length)
    var y_buffer = Span(unsafe_ptr=y_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var x_buffer, var y_buffer}:
        var idx = Int(idx0[0])
        var a = x_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var b = y_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, scalb(a, b))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = x_host[i]
            var y = y_host[i]
            var expected = libm_scalb(x, y)
            _check_exact(out_host[i], expected, x, y, "gpu", i)
            _check_exact(scalb(x, y), expected, x, y, "host", i)
    print(t"scalb {dtype}: gpu and host match libm")


def _fill_pairs[
    dtype: DType
](
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
    mode: PairMode,
    seed: UInt64,
) raises where dtype.is_floating_point():
    """Fills x and y with the `pair_inputs` for `mode`."""
    var pairs = pair_inputs(scalb_specials[dtype](), mode, seed, length)
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host:
        for i in range(length):
            x_host[i] = pairs[0][i]
            y_host[i] = pairs[1][i]


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime seed = 1234
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    var modes: List[PairMode] = [.random_bits, .integer_y]
    for i, mode in enumerate(modes):
        _fill_pairs(x, y, mode, UInt64(seed + i))
        run_scalb(ctx, x, y)


def test_scalb() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](ctx)
        _run_all[.float16](ctx)
        _run_all[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run_all[.float64](ctx)


def _run[
    dtype: DType, //, math_fn: UnaryFn
](ctx: DeviceContext, xs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Applies `math_fn` to `xs` on the GPU."""
    var n = len(xs)
    comptime pack_size = simd_width_of[dtype, target=ctx.target]()
    # Pad to a whole number of packs so every lane holds a real input.
    var padded = align_up(n, pack_size)
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = xs[i] if i < n else xs[0]
    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0])
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var out = List[Scalar[dtype]]()
    with out_device.map_to_host() as out_host:
        for i in range(n):
            out.append(out_host[i])
    return out^


def _check[
    dtype: DType
](
    xs: List[Scalar[dtype]],
    got: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
    name: String,
) raises:
    for i in range(len(xs)):
        assert_true(
            got[i].to_bits() == want[i].to_bits(),
            String(t"gpu {name}({xs[i]}) = {got[i]}, expected {want[i]}"),
        )


def _scalb_up[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime k = 149.0 if dtype == .float32 else 1074.0
    return scalb(x, SIMD[dtype, width](k))


def _scalb_inf[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return scalb(x, SIMD[dtype, width](inf[dtype]()))


def _scalb_neg_inf[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return scalb(x, SIMD[dtype, width](-inf[dtype]()))


def _scalb_unchanged[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return scalb(x, SIMD[dtype, width](0))


def _scalb_half[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return scalb(x, SIMD[dtype, width](-1))


def _scalb_exponent[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Uses `x` as the exponent of a fixed base."""
    return scalb(SIMD[dtype, width](3), x)


def _check_nan[
    dtype: DType
](xs: List[Scalar[dtype]], got: List[Scalar[dtype]], name: String) raises:
    for i in range(len(xs)):
        assert_true(
            isnan(got[i]),
            String(t"gpu {name}({xs[i]}) = {got[i]}, expected nan"),
        )


def test_subnormal_inputs_f32() raises:
    with DeviceContext() as ctx:
        var xs: List[Float32] = [f32_min, f32_min * 4, -f32_max_sub, 8.0]
        _check(
            xs,
            _run[_scalb_up](ctx, xs),
            [1.0, 4.0, -8388607.0, inf[.float32]()],
            "scalb",
        )
        # scalb(x, +-inf) must not read a subnormal x as 0 * inf.
        _check(
            xs,
            _run[_scalb_inf](ctx, xs),
            [
                inf[.float32](),
                inf[.float32](),
                -inf[.float32](),
                inf[.float32](),
            ],
            "scalb(x, +inf)",
        )
        _check(
            xs,
            _run[_scalb_neg_inf](ctx, xs),
            [0.0, 0.0, -0.0, 0.0],
            "scalb(x, -inf)",
        )


def test_subnormal_exponent_f32() raises:
    # A subnormal exponent is never integral, so the result is NaN even where
    # the exponent would flush to zero before `trunc` compares it.
    with DeviceContext() as ctx:
        var xs: List[Float32] = [f32_min, -f32_max_sub, f32_min * 4]
        _check_nan(xs, _run[_scalb_exponent](ctx, xs), "scalb(3, x)")


def test_subnormal_results_f32() raises:
    # Subnormal results are bit-exact on every target, Apple GPUs included.
    with DeviceContext() as ctx:
        var min_normal = Float32(from_bits=UInt32(0x00800000))  # 2^-126
        var xs: List[Float32] = [f32_min, min_normal, -f32_max_sub, 8.0]
        _check(
            xs,
            _run[_scalb_unchanged](ctx, xs),
            xs,
            "scalb(x, 0)",
        )
        _check(
            xs,
            _run[_scalb_half](ctx, xs),
            [
                0.0,
                Float32(from_bits=UInt32(0x00400000)),  # 2^-127
                Float32(from_bits=UInt32(0x80400000)),  # ties to even
                4.0,
            ],
            "scalb(x, -1)",
        )


def test_subnormal_inputs_f64() raises:
    comptime if not default_accelerator().is_apple_gpu():
        comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
        comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))
        with DeviceContext() as ctx:
            var xs: List[Float64] = [f64_min, -f64_max_sub, 8.0]
            _check(
                xs,
                _run[_scalb_up](ctx, xs),
                [1.0, -4503599627370495.0, inf[.float64]()],
                "scalb",
            )
            _check(
                xs,
                _run[_scalb_inf](ctx, xs),
                [inf[.float64](), -inf[.float64](), inf[.float64]()],
                "scalb(x, +inf)",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
