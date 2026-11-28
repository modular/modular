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
# Signed `MIN / -1` overflows. It must wrap like the compile-time evaluator:
#   MIN // -1 == MIN, MIN % -1 == 0, divmod(MIN, -1) == (MIN, 0).
# Before the fix these crash the process with SIGFPE (issue #7663).
# `black_box` keeps the compiler from folding the division at compile time.

from std.benchmark import black_box
from std.testing import assert_equal, TestSuite


def _check_min_by_neg_one[dtype: DType]() raises where dtype.is_signed():
    var lo = black_box(Scalar[dtype].MIN)
    var neg_one = black_box(Scalar[dtype](-1))
    assert_equal(lo // neg_one, lo)
    assert_equal(lo % neg_one, 0)
    var qr = divmod(lo, neg_one)
    assert_equal(qr[0], lo)
    assert_equal(qr[1], 0)


def test_min_by_neg_one_int8() raises:
    _check_min_by_neg_one[DType.int8]()


def test_min_by_neg_one_int16() raises:
    _check_min_by_neg_one[DType.int16]()


def test_min_by_neg_one_int32() raises:
    _check_min_by_neg_one[DType.int32]()


def test_min_by_neg_one_int64() raises:
    _check_min_by_neg_one[DType.int64]()


def test_min_by_neg_one_int() raises:
    var lo = black_box(Int.MIN)
    var neg_one = black_box(Int(-1))
    assert_equal(lo // neg_one, lo)
    assert_equal(lo % neg_one, 0)
    var qr = divmod(lo, neg_one)
    assert_equal(qr[0], lo)
    assert_equal(qr[1], 0)


def test_min_by_neg_one_matches_compile_time() raises:
    comptime q = Int.MIN // -1
    comptime r = Int.MIN % -1
    assert_equal(black_box(Int.MIN) // black_box(Int(-1)), q)
    assert_equal(black_box(Int.MIN) % black_box(Int(-1)), r)


def test_neighbors_unchanged() raises:
    # Values next to the overflow case must keep their normal results.
    var neg_one = black_box(Int(-1))
    var one = black_box(Int(1))
    assert_equal(black_box(Int(5)) // neg_one, -5)
    assert_equal(black_box(Int(5)) % neg_one, 0)
    assert_equal(black_box(Int(-7)) // neg_one, 7)
    assert_equal(neg_one // neg_one, 1)
    assert_equal(neg_one % neg_one, 0)
    assert_equal(black_box(Int.MIN) // one, Int.MIN)
    assert_equal(black_box(Int.MIN) % one, 0)
    assert_equal(black_box(Int.MIN + 1) // neg_one, Int.MAX)
    assert_equal(black_box(Int.MIN + 1) % neg_one, 0)
    assert_equal(black_box(Int.MAX) // neg_one, -Int.MAX)
    assert_equal(black_box(Int.MIN) // black_box(Int(2)), Int.MIN // 2)
    assert_equal(black_box(Int.MIN) % black_box(Int(2)), 0)


def test_neighbors_unchanged_int8() raises:
    var neg_one = black_box(Int8(-1))
    assert_equal(black_box(Int8(-127)) // neg_one, 127)
    assert_equal(black_box(Int8(-127)) % neg_one, 0)
    assert_equal(black_box(Int8(127)) // neg_one, -127)
    assert_equal(black_box(Int8(-128)) // black_box(Int8(1)), -128)
    assert_equal(black_box(Int8(-128)) // black_box(Int8(2)), -64)


def test_zero_divisor_still_guarded() raises:
    # The existing zero guard must keep working next to the new case.
    assert_equal(black_box(Int.MIN) // black_box(Int(0)), 0)
    assert_equal(black_box(Int.MIN) % black_box(Int(0)), 0)
    assert_equal(black_box(Int32(7)) // black_box(Int32(0)), 0)


def test_simd_int32_mixed_lanes() raises:
    var x = black_box(SIMD[DType.int32, 4](10, Int32.MIN, -7, 0))
    var y = black_box(SIMD[DType.int32, 4](-1, -1, 2, -1))
    assert_equal(x // y, SIMD[DType.int32, 4](-10, Int32.MIN, -4, 0))
    assert_equal(x % y, SIMD[DType.int32, 4](0, 0, 1, 0))
    var qr = divmod(x, y)
    assert_equal(qr[0], SIMD[DType.int32, 4](-10, Int32.MIN, -4, 0))
    assert_equal(qr[1], SIMD[DType.int32, 4](0, 0, 1, 0))


def test_simd_int8_all_lanes_overflow() raises:
    var x = black_box(SIMD[DType.int8, 8](Int8.MIN))
    var y = black_box(SIMD[DType.int8, 8](-1))
    assert_equal(x // y, SIMD[DType.int8, 8](Int8.MIN))
    assert_equal(x % y, SIMD[DType.int8, 8](0))


def test_simd_int64_overflow_and_zero_lanes() raises:
    var x = black_box(SIMD[DType.int64, 4](Int64.MIN, Int64.MIN, 5, Int64.MIN))
    var y = black_box(SIMD[DType.int64, 4](-1, 0, -1, 1))
    assert_equal(x // y, SIMD[DType.int64, 4](Int64.MIN, 0, -5, Int64.MIN))
    assert_equal(x % y, SIMD[DType.int64, 4](0, 0, 0, 0))


def test_unsigned_unaffected() raises:
    # `-1` for unsigned is the maximum value: no special handling.
    var m = black_box(UInt8(255))
    assert_equal(black_box(UInt8(255)) // m, 1)
    assert_equal(black_box(UInt8(0)) // m, 0)
    assert_equal(black_box(UInt8(100)) % m, 100)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
