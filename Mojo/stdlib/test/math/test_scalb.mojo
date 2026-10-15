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

from std.ffi import _external_call_const
from std.math import inf, isinf, isnan, nan, scalb, trunc
from std.math.math import _narrow_f32_to_bf16_bits, _scale_down_bits
from std.sys import CompilationTarget, size_of
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import FPUtils, max_finite

comptime length = 1 << 12


def _scalb_ref[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Returns fdlibm/glibc `scalb` built on libm `ldexp`.

    macOS libm has no `scalbf`, and its `scalb` does not return NaN for a
    non-integral exponent, so it is not a usable reference there. Scaling a
    float32 or narrower value is exact in float64, so the rounded `ldexp`
    result is correct. Narrower types round through float32 like `scalbf`.
    """
    # Clamping the exponent this far out saturates `ldexp` for every dtype
    # without overflowing `Int32`.
    comptime exp_limit = 65000.0
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
        var n = Int32(y64.clamp(-exp_limit, exp_limit))
        r = _external_call_const["ldexp", Float64](x64, n)
    comptime if size_of[dtype]() < size_of[DType.float32]():
        return r.cast[.float32]().cast[dtype]()
    return r.cast[dtype]()


def _libm[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    comptime if not CompilationTarget.is_linux():
        return _scalb_ref(x, y)
    comptime if dtype == .float64:
        return _external_call_const["scalb", Scalar[dtype]](x, y)
    comptime if dtype == .float32:
        return _external_call_const["scalbf", Scalar[dtype]](x, y)
    return _external_call_const["scalbf", Float32](
        x.cast[.float32](), y.cast[.float32]()
    ).cast[dtype]()


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
        comptime nbits = 8 * size_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        return Float64(self.next() >> 11) / Float64(1 << 53)


def _specials[dtype: DType]() -> List[Scalar[dtype]]:
    comptime IT = Scalar[FPUtils[dtype].integral_type]
    var big = max_finite[dtype]()
    var tiny = Scalar[dtype](from_bits=IT(1))
    var min_normal = Scalar[dtype](
        from_bits=IT(1 << FPUtils[dtype].mantissa_width())
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


def _inputs[
    dtype: DType
](mode: Int, seed: UInt64) -> Tuple[List[Scalar[dtype]], List[Scalar[dtype]]]:
    """Special-value pairs, then `mode`-specific inputs.

    mode 0: random bit patterns for both operands.
    mode 1: x random bits, y a small integer (some non-integral, some huge).
    """
    var rng = _Rng(seed)
    var sp = _specials[dtype]()
    var xs = List[Scalar[dtype]]()
    var ys = List[Scalar[dtype]]()
    for a in sp:
        for b in sp:
            xs.append(a)
            ys.append(b)
    var i = len(xs)
    while i < length:
        var x = rng.bits[dtype]()
        var y: Scalar[dtype]
        if mode == 0:
            y = rng.bits[dtype]()
        else:
            comptime span = 2 * (
                FPUtils[dtype].exponent_bias()
                + FPUtils[dtype].mantissa_width()
                + 4
            )
            var n = Float64(Int(rng.uniform() * Float64(span)) - span // 2)
            if i % 17 == 0:
                n += 0.5
            elif i % 29 == 0:
                n *= 1e6
            y = Scalar[dtype](n)
        xs.append(x)
        ys.append(y)
        i += 1
    return (xs^, ys^)


def _run[dtype: DType]() raises where dtype.is_floating_point():
    comptime width = 4
    for mode in range(2):
        var inputs = _inputs[dtype](mode, UInt64(1234 + mode))
        ref xs = inputs[0]
        ref ys = inputs[1]
        for base in range(0, length, width):
            var xv = SIMD[dtype, width]()
            var yv = SIMD[dtype, width]()
            comptime for j in range(width):
                xv[j] = xs[base + j]
                yv[j] = ys[base + j]
            var got = scalb(xv, yv)
            comptime for j in range(width):
                var x = xv[j]
                var y = yv[j]
                var expected = _libm(x, y)
                # scalb is exact, so it must match bit for bit (including the
                # sign of zero).
                var ok: Bool
                if isnan(expected):
                    ok = isnan(got[j])
                else:
                    ok = got[j].to_bits() == expected.to_bits()
                if not ok:
                    raise Error(
                        t"scalb {dtype} ({x}, {y}) got {got[j]} expected"
                        t" {expected}"
                    )


def test_float32() raises:
    _run[.float32]()


def test_float64() raises:
    _run[.float64]()


def test_float16() raises:
    _run[.float16]()


def test_bfloat16() raises:
    _run[.bfloat16]()


def test_narrow_exponent_promotion() raises:
    # Exponents outside int16 must be handled after promotion to float32.
    comptime for dtype in [DType.float16, DType.bfloat16]:
        comptime assert dtype.is_floating_point()
        var x = SIMD[dtype, 4](1.0, -1.0, -0.0, 0.0)
        var positive = scalb(x, SIMD[dtype, 4](40000))
        var negative = scalb(x, SIMD[dtype, 4](-40000))
        var want_positive = SIMD[dtype, 4](
            inf[dtype](), -inf[dtype](), -0.0, 0.0
        )
        var want_negative = SIMD[dtype, 4](0.0, -0.0, -0.0, 0.0)
        comptime for lane in range(4):
            assert_equal(
                positive[lane].to_bits(), want_positive[lane].to_bits()
            )
            assert_equal(
                negative[lane].to_bits(), want_negative[lane].to_bits()
            )


def test_scale_down_bits() raises:
    # Correct rounding, including ties and subnormal results, against `ldexp`
    # in float64 (exact for float32 operands) rounded once to float32.
    comptime shift = 24
    var state = UInt32(12345)
    for i in range(600):
        state = state * 1664525 + 1013904223
        var bits = state
        if i % 3 == 0:
            bits = (bits & 0x807FFFFF) | (UInt32(1 + (i // 3) % 40) << 23)
        elif i % 3 == 1:
            # Low mantissa bits clear so some results land exactly on a tie.
            bits = (bits & 0x80FFF000) | (UInt32(1 + i % 60) << 23)
        else:
            bits = (bits & 0x807FFFFF) | (UInt32(100 + i % 150) << 23)
        var y = Float32(from_bits=bits)
        var want = _external_call_const["ldexp", Float64](
            y.cast[.float64](), Int32(-shift)
        ).cast[.float32]()
        var got = _scale_down_bits[shift](y)
        assert_equal(got.to_bits(), want.to_bits())


def test_narrow_f32_to_bf16_bits() raises:
    var state = UInt32(777)
    for i in range(600):
        state = state * 1664525 + 1013904223
        var bits = state
        if i % 4 == 0:
            bits = bits & 0x807FFFFF  # float32 subnormals
        elif i % 4 == 1:
            bits = (bits & 0xFFFF0000) | 0x8000  # exact ties
        var x = Float32(from_bits=bits)
        var got = _narrow_f32_to_bf16_bits(x)
        if isnan(x):
            assert_true(isnan(got))
        else:
            assert_equal(got.to_bits(), x.cast[.bfloat16]().to_bits())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
