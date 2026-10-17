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

from std.simd import RaisingSIMD, SaturatingSIMD, TrappingSIMD
from std.testing import assert_equal, assert_true, TestSuite


def test_raising_no_overflow() raises:
    comptime R = RaisingSIMD[.int8, 4]
    var a = R(SIMD[.int8, 4](1, 2, 3, 4))
    var b = R(SIMD[.int8, 4](10, 20, 30, 31))
    assert_equal((a + b).value, SIMD[.int8, 4](11, 22, 33, 35))
    assert_equal((b - a).value, SIMD[.int8, 4](9, 18, 27, 27))
    assert_equal((a * b).value, SIMD[.int8, 4](10, 40, 90, 124))


def test_raising_add_overflow() raises:
    comptime R = RaisingSIMD[.int8, 4]
    var a = R(SIMD[.int8, 4](127, 0, -128, 1))
    var b = R(SIMD[.int8, 4](1, 0, -1, 1))
    var raised = False
    try:
        _ = a + b
    except e:
        raised = True
        assert_equal(e.overflowed, SIMD[.bool, 4](True, False, True, False))
        assert_equal(e.overflow_bits, SIMD[.int8, 4](-128, 0, 127, 2))
    assert_true(raised)


def test_raising_unsigned() raises:
    comptime R = RaisingSIMD[.uint8, 2]
    var a = R(SIMD[.uint8, 2](0, 5))
    var b = R(SIMD[.uint8, 2](1, 1))
    var raised = False
    try:
        _ = a - b
    except e:
        raised = True
        assert_equal(e.overflowed, SIMD[.bool, 2](True, False))
        assert_equal(e.overflow_bits, SIMD[.uint8, 2](255, 4))
    assert_true(raised)


def test_raising_scalar_mul() raises:
    comptime R = RaisingSIMD[.int64, 1]
    var raised = False
    try:
        _ = R(Int64.MAX) * R(2)
    except e:
        raised = True
        assert_true(e.overflowed)
    assert_true(raised)
    assert_equal((R(-3) * R(4)).value, -12)


def test_saturating_mul_256() raises:
    # 256-bit lanes have no wider type and use the division-based check.
    comptime S = SaturatingSIMD[.int256, 4]
    comptime T = SIMD[.int256, 4]
    var a = S(T(T.MAX[0], -1, 3, 0))
    var b = S(T(2, T.MIN[0], -4, T.MIN[0]))
    assert_equal((a * b).value, T(T.MAX[0], T.MAX[0], -12, 0))

    comptime U = SaturatingSIMD[.uint256, 2]
    comptime V = SIMD[.uint256, 2]
    var c = U(V(V.MAX[0], 7))
    var d = U(V(2, 6))
    assert_equal((c * d).value, V(V.MAX[0], 42))


def test_saturating_signed() raises:
    comptime S = SaturatingSIMD[.int8, 4]
    var a = S(SIMD[.int8, 4](120, -120, 5, -100))
    var b = S(SIMD[.int8, 4](10, -10, 5, 2))
    assert_equal((a + b).value, SIMD[.int8, 4](127, -128, 10, -98))
    assert_equal((a - b).value, SIMD[.int8, 4](110, -110, 0, -102))
    assert_equal((a * b).value, SIMD[.int8, 4](127, 127, 25, -128))


def test_saturating_unsigned() raises:
    comptime S = SaturatingSIMD[.uint8, 4]
    var a = S(SIMD[.uint8, 4](250, 3, 16, 0))
    var b = S(SIMD[.uint8, 4](10, 5, 16, 0))
    assert_equal((a + b).value, SIMD[.uint8, 4](255, 8, 32, 0))
    assert_equal((a - b).value, SIMD[.uint8, 4](240, 0, 0, 0))
    assert_equal((a * b).value, SIMD[.uint8, 4](255, 15, 255, 0))


def test_trapping_no_overflow() raises:
    comptime T = TrappingSIMD[.int32, 4]
    var a = T(SIMD[.int32, 4](1, -2, 3, -4))
    var b = T(SIMD[.int32, 4](5, 6, -7, -8))
    assert_equal((a + b).value, SIMD[.int32, 4](6, 4, -4, -12))
    assert_equal((a - b).value, SIMD[.int32, 4](-4, -8, 10, 4))
    assert_equal((a * b).value, SIMD[.int32, 4](5, -12, -21, 32))


def test_simd_methods() raises:
    var a = SIMD[.int8, 4](120, -120, 5, -100)
    var b = SIMD[.int8, 4](10, -10, 5, 2)

    assert_equal(a.saturating_add(b), SIMD[.int8, 4](127, -128, 10, -98))
    assert_equal(a.saturating_sub(b), SIMD[.int8, 4](110, -110, 0, -102))
    assert_equal(a.saturating_mul(b), SIMD[.int8, 4](127, 127, 25, -128))
    assert_equal((a.saturating() + b.saturating()).value, a.saturating_add(b))

    var c = SIMD[.int8, 4](1, 2, 3, 4)
    assert_equal(c.trapping_add(c), SIMD[.int8, 4](2, 4, 6, 8))
    assert_equal(c.trapping_sub(c), SIMD[.int8, 4](0))
    assert_equal(c.trapping_mul(c), SIMD[.int8, 4](1, 4, 9, 16))
    assert_equal((c.trapping() * c.trapping()).value, c.trapping_mul(c))

    assert_equal(c.raising_add(c), SIMD[.int8, 4](2, 4, 6, 8))
    assert_equal(c.raising_sub(c), SIMD[.int8, 4](0))
    assert_equal(c.raising_mul(c), SIMD[.int8, 4](1, 4, 9, 16))
    assert_equal((c.raising() + c.raising()).value, c.raising_add(c))

    var raised = False
    try:
        _ = a.raising_add(b)
    except e:
        raised = True
        assert_equal(e.overflowed, SIMD[.bool, 4](True, True, False, False))
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
