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

from std.math import isnan, remainder
from std.math.math import (
    _is_zero_bits,
    _narrow_f32_to_bf16_bits,
    _scale_down_bits,
    _scale_up_subnormal,
)
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from test_utils.math import (
    PairMode,
    SplitMix64,
    libm_call_binary,
    pair_inputs,
    remainder_specials,
)

comptime length = 1 << 12


def _run[dtype: DType, width: Int]() raises where dtype.is_floating_point():
    # remainder is exact, so it must match libm bit for bit, including the
    # sign of a zero result.
    var modes: List[PairMode] = [
        .random_bits,
        .integer_y,
        .small_quotients,
        .near_multiples,
    ]
    for i, mode in enumerate(modes):
        var inputs = pair_inputs(
            remainder_specials[dtype](), mode, UInt64(1234 + i), length
        )
        ref xs = inputs[0]
        ref ys = inputs[1]
        for base in range(0, length, width):
            var xv = SIMD[dtype, width]()
            var yv = SIMD[dtype, width]()
            comptime for j in range(width):
                xv[j] = xs[base + j]
                yv[j] = ys[base + j]
            var got = remainder(xv, yv)
            comptime for j in range(width):
                var x = xv[j]
                var y = yv[j]
                var expected = libm_call_binary["remainderf", "remainder"](x, y)
                var ok: Bool
                if isnan(expected):
                    ok = isnan(got[j])
                else:
                    ok = got[j].to_bits() == expected.to_bits()
                if not ok:
                    raise Error(
                        t"remainder {dtype} ({x}, {y}) got {got[j]} expected"
                        t" {expected}"
                    )


def _f32_from_bits(bits: UInt64) -> Float32:
    return Float32(from_bits=bits.cast[DType.uint32]())


def test_is_zero_bits() raises:
    var rng = SplitMix64(7)
    for _ in range(512):
        var x = rng.bits[.float32]()
        var m = x.to_bits() & 0x7FFFFFFF
        assert_equal(_is_zero_bits(x), m == 0)
    assert_true(_is_zero_bits(Float32(0.0)))
    assert_true(_is_zero_bits(Float32(-0.0)))
    assert_false(_is_zero_bits(_f32_from_bits(1)))
    assert_false(_is_zero_bits(_f32_from_bits(0x80000001)))


def _check_scale_up[shift: Int]() raises:
    var rng = SplitMix64(11 + UInt64(shift))
    for i in range(512):
        var bits = rng.next() & 0x807FFFFF
        if i == 0:
            bits = 1
        elif i == 1:
            bits = 0x007FFFFF
        elif bits & 0x7FFFFF == 0:
            bits |= 1
        var x = _f32_from_bits(bits)
        var expected = (Float64(x) * Float64(2) ** Float64(shift)).cast[
            DType.float32
        ]()
        var got = _scale_up_subnormal[shift](x)
        assert_equal(got.to_bits(), expected.to_bits())


def test_scale_up_subnormal() raises:
    _check_scale_up[23]()
    _check_scale_up[46]()
    _check_scale_up[100]()


def _check_scale_down[shift: Int]() raises:
    var rng = SplitMix64(13 + UInt64(shift))
    for i in range(1024):
        var bits = rng.next() & 0xFFFFFFFF
        # Zero low bits make exact ties for the subnormal rounding common.
        bits &= ~((UInt64(1) << (rng.next() % 24)) - 1)
        var e = (bits >> 23) & 0xFF
        # Keep y finite and normal; exponents near 1 + shift hit the
        # subnormal range, the rest the normal and underflow-to-zero ranges.
        if i % 2 == 0:
            e = UInt64(max(1 + shift - 26, 1)) + rng.next() % 28
        elif e == 0 or e == 255:
            e = 100
        bits = (bits & 0x807FFFFF) | (e << 23)
        var y = _f32_from_bits(bits)
        var expected = (Float64(y) * Float64(2) ** Float64(-shift)).cast[
            DType.float32
        ]()
        var got = _scale_down_bits[shift](y)
        assert_equal(got.to_bits(), expected.to_bits())


def test_scale_down_bits() raises:
    _check_scale_down[1]()
    _check_scale_down[23]()
    _check_scale_down[46]()
    _check_scale_down[100]()


def test_narrow_f32_to_bf16_bits() raises:
    var rng = SplitMix64(17)
    for i in range(2048):
        var bits = rng.next() & 0xFFFFFFFF
        if i % 2 == 0:
            # Bring the exponent to the subnormal bfloat16 range and make
            # ties common.
            bits = (bits & 0x807FFFFF) | ((rng.next() % 2) << 23)
            bits &= ~((UInt64(1) << (rng.next() % 17)) - 1)
        if i % 8 == 1:
            bits = (bits & 0x807FFFFF) | 0x7F800000
        var x = _f32_from_bits(bits)
        var expected = x.cast[DType.bfloat16]()
        var got = _narrow_f32_to_bf16_bits(x)
        if isnan(expected):
            assert_true(isnan(got))
        else:
            assert_equal(got.to_bits(), expected.to_bits())


def test_float32() raises:
    _run[.float32, 1]()
    _run[.float32, 4]()


def test_float64() raises:
    _run[.float64, 1]()
    _run[.float64, 4]()


def test_float16() raises:
    _run[.float16, 1]()
    _run[.float16, 4]()


def test_bfloat16() raises:
    _run[.bfloat16, 1]()
    _run[.bfloat16, 4]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
