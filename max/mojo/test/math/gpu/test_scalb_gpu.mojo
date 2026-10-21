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

from std.ffi import _external_call_const
from std.math import align_up, inf, isinf, isnan, nan, scalb, trunc
from std.sys import (
    CompilationTarget,
    simd_width_of,
    size_of,
)
from std.utils.numerics import FPUtils, max_finite

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from std.testing import TestSuite, assert_true

from std.utils.coord import Coord

comptime length = 8192

comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))


def _scalb_ref[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Returns fdlibm `scalb` built on libm `ldexp`, for hosts without `scalbf`.

    Scaling a float32 or narrower value is exact in float64, so the rounded
    `ldexp` result is correct. Narrower types round through float32 like
    `scalbf`.
    """
    var x64 = x.cast[.float64]()
    var y64 = y.cast[.float64]()
    var r: Float64
    if isnan(x64) or isnan(y64):
        r = x64 * y64
    elif isinf(y64):
        r = x64 * y64 if y64 > 0 else x64 / -y64
    elif trunc(y64) != y64:
        r = nan[.float64]()
    else:
        # Clamping this far out saturates `ldexp` for every dtype without
        # overflowing `Int32`.
        var n = Int32(y64.clamp(-65000.0, 65000.0))
        r = _external_call_const["ldexp", Float64](x64, n)
    comptime if size_of[dtype]() < size_of[DType.float32]():
        return r.cast[.float32]().cast[dtype]()
    return r.cast[dtype]()


def _libm[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Calls host libm directly; f16/bf16 round the f32 libm result."""
    comptime if not CompilationTarget.is_linux():
        return _scalb_ref(x, y)
    comptime if dtype == .float64:
        return _external_call_const["scalb", Scalar[dtype]](x, y)
    comptime if dtype == .float32:
        return _external_call_const["scalbf", Scalar[dtype]](x, y)
    return _external_call_const["scalbf", Float32](
        x.cast[.float32](), y.cast[.float32]()
    ).cast[dtype]()


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
            var expected = _libm(x, y)
            _check_exact(out_host[i], expected, x, y, "gpu", i)
            _check_exact(scalb(x, y), expected, x, y, "host", i)
    print(t"scalb {dtype}: gpu and host match libm")


struct _Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        # splitmix64
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def bits[dtype: DType](mut self) -> Scalar[dtype]:
        """Returns a uniformly random bit pattern of `dtype`."""
        comptime nbits = 8 * size_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        return Float64(self.next() >> 11) / Float64(1 << 53)


def _specials[dtype: DType]() -> List[Scalar[dtype]]:
    var big = max_finite[dtype]()
    var tiny = Scalar[dtype](from_bits=Scalar[FPUtils[dtype].integral_type](1))
    var min_normal = Scalar[dtype](
        from_bits=Scalar[FPUtils[dtype].integral_type](
            1 << FPUtils[dtype].mantissa_width()
        )
    )
    return [
        0,
        -0.0,
        1,
        -1,
        0.5,
        1.5,
        2.5,
        -2.5,
        3,
        -7,
        0.75,
        10,
        big,
        -big,
        tiny,
        -tiny,
        min_normal,
        -min_normal,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
    ]


def _fill_pairs[
    dtype: DType
](
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
    mode: Int,
    seed: UInt64,
) raises:
    """Fills x and y with special-value pairs, then `mode`-specific inputs.

    mode 0: random bit patterns for both operands.
    mode 1: x random bits, y a small integer (some non-integral, some huge).
    """
    var rng = _Rng(seed)
    var sp = _specials[dtype]()
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host:
        var i = 0
        for a in sp:
            for b in sp:
                x_host[i] = a
                y_host[i] = b
                i += 1
        while i < length:
            var x = rng.bits[dtype]()
            var y: Scalar[dtype]
            if mode == 0:
                y = rng.bits[dtype]()
            else:
                var r = rng.uniform()
                comptime span = 2 * (
                    FPUtils[dtype].exponent_bias()
                    + FPUtils[dtype].mantissa_width()
                    + 4
                )
                var n = Float64(Int(r * Float64(span)) - span // 2)
                if i % 17 == 0:
                    n += 0.5
                elif i % 29 == 0:
                    n *= 1e6
                y = Scalar[dtype](n)
            x_host[i] = x
            y_host[i] = y
            i += 1


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime num_fill_modes = 2
    comptime seed = 1234
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    for mode in range(num_fill_modes):
        _fill_pairs(x, y, mode, UInt64(seed + mode))
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
