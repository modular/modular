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
from std.testing import TestSuite

from test_utils.math import (
    PairMode,
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
