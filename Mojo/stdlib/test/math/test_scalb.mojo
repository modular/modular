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

from std.math import inf, isnan, scalb
from std.testing import TestSuite, assert_equal

from test_utils.math import PairMode, libm_scalb, pair_inputs, scalb_specials

comptime length = 1 << 12


def _run[dtype: DType]() raises where dtype.is_floating_point():
    comptime width = 4
    var modes: List[PairMode] = [.random_bits, .integer_y]
    for i, mode in enumerate(modes):
        var inputs = pair_inputs(
            scalb_specials[dtype](), mode, UInt64(1234 + i), length
        )
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
                var expected = libm_scalb(x, y)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
