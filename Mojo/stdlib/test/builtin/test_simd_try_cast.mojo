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

from std.iter import filter
from std.itertools import product
from std.testing import assert_equal, assert_raises, TestSuite
from test_utils import check_write_to

# The oracle compares in `Int256`, so the 256-bit dtypes are tested by hand.
comptime INT_DTYPES = [
    DType.int8,
    DType.uint8,
    DType.int16,
    DType.uint16,
    DType.int32,
    DType.uint32,
    DType.int64,
    DType.uint64,
    DType.int,
    DType.uint,
    DType.int128,
    DType.uint128,
]


def in_range[dtype: DType](value: Int256) -> Bool:
    return Int256(Scalar[dtype].MIN) <= value and value <= Int256(
        Scalar[dtype].MAX
    )


def cast_or_none[
    src: DType, width: SIMDLength, //, dst: DType
](value: SIMD[src, width]) -> Optional[SIMD[dst, width]]:
    comptime assert src.is_integral() and dst.is_integral()
    try:
        return value.try_cast[dst]()
    except:
        return None


def check_try_cast[src: DType, dst: DType, width: Int](value: Int256) raises:
    # Each lane holds `value` in turn, with zeros elsewhere, so a reduction
    # that skips a lane fails.
    var expected = in_range[dst](value)
    for lane in range(width):
        var input = SIMD[src, width](0)
        input[lane] = value.cast[src]()
        var actual = cast_or_none[dst](input)
        var correct: Bool
        match actual:
            case .Some(result):
                correct = (
                    expected and result.cast[.int256]() == input.cast[.int256]()
                )
            case .None:
                correct = not expected
        if not correct:
            var outcome = String(t"return {input}") if expected else "raise"
            raise Error(
                t"SIMD[{src}, {width}]({input}).try_cast[{dst}]()",
                t" should {outcome}, got {actual}",
            )


def check_boundaries[src: DType, dst: DType, width: Int]() raises:
    var candidates = [
        Int256(0),
        1,
        -1,
        Int256(Scalar[src].MIN),
        Int256(Scalar[src].MAX),
        Int256(Scalar[dst].MIN) - 1,
        Int256(Scalar[dst].MIN),
        Int256(Scalar[dst].MAX),
        Int256(Scalar[dst].MAX) + 1,
    ]
    for value in filter(candidates, in_range[src]):
        check_try_cast[src, dst, width](value)


def test_try_cast_boundaries() raises:
    comptime for src, dst in product(INT_DTYPES, INT_DTYPES):
        check_boundaries[src, dst, 1]()
        check_boundaries[src, dst, 4]()


def test_try_cast_wide_vectors() raises:
    check_boundaries[.int8, .uint8, 64]()
    check_boundaries[.int16, .uint8, 64]()
    check_boundaries[.uint64, .int32, 16]()
    check_boundaries[.int, .uint, 8]()


def test_try_cast_256_bit() raises:
    assert_equal(Int256.MAX.try_cast[.uint256](), UInt256(Int256.MAX))
    assert_equal(UInt128.MAX.try_cast[.int256](), Int256(UInt128.MAX))
    assert_equal(Int256(-128).try_cast[.int8](), -128)
    assert_equal(UInt256(UInt64.MAX).try_cast[.uint64](), UInt64.MAX)
    with assert_raises():
        _ = UInt256.MAX.try_cast[.int256]()
    with assert_raises():
        _ = (UInt256(Int256.MAX) + 1).try_cast[.int256]()
    with assert_raises():
        _ = Int256(-1).try_cast[.uint256]()
    with assert_raises():
        _ = (UInt256(UInt64.MAX) + 1).try_cast[.uint64]()
    with assert_raises():
        _ = Int256(128).try_cast[.int8]()
    with assert_raises():
        _ = Int256(-129).try_cast[.int8]()
    with assert_raises():
        _ = UInt256.MAX.try_cast[.int8]()
    with assert_raises():
        _ = Int8(-1).try_cast[.uint256]()


def test_error() raises:
    try:
        _ = Int(256).try_cast[.uint8]()
    except e:
        check_write_to(
            e,
            expected="out of range integral type conversion attempted",
            is_repr=False,
        )
        return
    raise Error("Int(256).try_cast[.uint8]() didn't raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
