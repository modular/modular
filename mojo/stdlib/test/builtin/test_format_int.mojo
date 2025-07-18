# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from builtin.format_int import _format_int
from testing import assert_equal


fn test_format_int() raises:
    assert_equal(_format_int[DType.index](123), "123")
    assert_equal(_format_int[DType.index](4, 2), "100")
    assert_equal(_format_int[DType.index](255, 2), "11111111")
    assert_equal(_format_int[DType.index](254, 2), "11111110")
    assert_equal(_format_int[DType.index](255, 36), "73")

    assert_equal(_format_int[DType.index](-123, 10), "-123")
    assert_equal(_format_int[DType.index](-999_999_999, 10), "-999999999")

    # i64
    assert_equal(_format_int(Int64.MAX_FINITE, 10), "9223372036854775807")
    assert_equal(_format_int(Int64.MIN_FINITE, 10), "-9223372036854775808")
    assert_equal(_format_int(Int64.MAX_FINITE, 2), "1" * 63)
    assert_equal(_format_int(Int64.MIN_FINITE, 2), "-1" + "0" * 63)

    # i128
    alias int128_max = Int128(UInt128(1) << 127) - 1
    alias int128_min = Int128(UInt128(1) << 127)
    assert_equal(
        _format_int(int128_max, 10),
        "170141183460469231731687303715884105727",
    )
    assert_equal(
        _format_int(int128_min, 10),
        "-170141183460469231731687303715884105728",
    )
    assert_equal(_format_int(int128_max, 2), "1" * 127)
    assert_equal(_format_int(int128_min, 2), "-1" + "0" * 127)

    # i256
    alias int256_max = Int256(UInt256(1) << 255) - 1
    alias int256_min = Int256(UInt256(1) << 255)
    assert_equal(
        _format_int(int256_max, 10),
        "57896044618658097711785492504343953926634992332820282019728792003956564819967",
    )
    assert_equal(
        _format_int(int256_min, 10),
        "-57896044618658097711785492504343953926634992332820282019728792003956564819968",
    )
    assert_equal(_format_int(int256_max, 2), "1" * 255)
    assert_equal(_format_int(int256_min, 2), "-1" + "0" * 255)


fn test_hex() raises:
    assert_equal(hex(0), "0x0")
    assert_equal(hex(1), "0x1")
    assert_equal(hex(5), "0x5")
    assert_equal(hex(10), "0xa")
    assert_equal(hex(255), "0xff")
    assert_equal(hex(128), "0x80")
    assert_equal(hex(1 << 16), "0x10000")

    # Max and min i64 values in base 16
    assert_equal(hex(Int64.MAX_FINITE), "0x7fffffffffffffff")

    # Negative values
    assert_equal(hex(-0), "0x0")
    assert_equal(hex(-1), "-0x1")
    assert_equal(hex(-10), "-0xa")
    assert_equal(hex(-255), "-0xff")

    assert_equal(hex(Int64.MIN_FINITE), "-0x8000000000000000")

    # SIMD values
    assert_equal(hex(Int32(45)), "0x2d")
    assert_equal(hex(Int8(2)), "0x2")
    assert_equal(hex(Int8(-2)), "-0x2")
    assert_equal(hex(Scalar[DType.bool](True)), "0x1")
    assert_equal(hex(False), "0x0")


@fieldwise_init
struct Ind(Intable):
    fn __int__(self) -> Int:
        return 1


def test_bin_scalar():
    assert_equal(bin(Int8(2)), "0b10")
    assert_equal(bin(Int32(123)), "0b1111011")
    assert_equal(bin(Int32(-123)), "-0b1111011")
    assert_equal(bin(Scalar[DType.bool](True)), "0b1")
    assert_equal(bin(Scalar[DType.bool](False)), "0b0")


def test_bin_int():
    assert_equal(bin(0), "0b0")
    assert_equal(bin(1), "0b1")
    assert_equal(bin(-1), "-0b1")
    assert_equal(bin(4), "0b100")
    assert_equal(bin(Int(-4)), "-0b100")
    assert_equal(bin(389703), "0b1011111001001000111")
    assert_equal(bin(-10), "-0b1010")


def test_bin_bool():
    assert_equal(bin(True), "0b1")
    assert_equal(bin(False), "0b0")


def test_oct_scalar():
    assert_equal(oct(Int32(234)), "0o352")
    assert_equal(oct(Int32(-23)), "-0o27")
    assert_equal(oct(Int32(0)), "0o0")
    assert_equal(oct(Scalar[DType.bool](True)), "0o1")
    assert_equal(oct(Scalar[DType.bool](False)), "0o0")


def test_oct_int():
    assert_equal(oct(768), "0o1400")
    assert_equal(oct(-12), "-0o14")
    assert_equal(oct(23623564), "0o132073614")
    assert_equal(oct(0), "0o0")
    assert_equal(oct(1), "0o1")
    assert_equal(oct(Int(7658)), "0o16752")


def test_oct_bool():
    assert_equal(oct(True), "0o1")
    assert_equal(oct(False), "0o0")


def test_intable():
    assert_equal(bin(Ind()), "0b1")
    assert_equal(hex(Ind()), "0x1")
    assert_equal(oct(Ind()), "0o1")


def test_different_prefix():
    assert_equal(bin(Int8(1), prefix="binary"), "binary1")
    assert_equal(hex(Int8(1), prefix="hexadecimal"), "hexadecimal1")
    assert_equal(oct(Int8(1), prefix="octal"), "octal1")

    assert_equal(bin(0, prefix="binary"), "binary0")
    assert_equal(hex(0, prefix="hexadecimal"), "hexadecimal0")
    assert_equal(oct(0, prefix="octal"), "octal0")

    assert_equal(bin(Ind(), prefix="I'mAnIndexer!"), "I'mAnIndexer!1")
    assert_equal(hex(Ind(), prefix="I'mAnIndexer!"), "I'mAnIndexer!1")
    assert_equal(oct(Ind(), prefix="I'mAnIndexer!"), "I'mAnIndexer!1")

    assert_equal(bin(Scalar[DType.bool](True), prefix="test"), "test1")
    assert_equal(hex(Scalar[DType.bool](True), prefix="test"), "test1")
    assert_equal(oct(Scalar[DType.bool](True), prefix="test"), "test1")


def main():
    test_format_int()
    test_hex()
    test_bin_scalar()
    test_bin_int()
    test_bin_bool()
    test_intable()
    test_oct_scalar()
    test_oct_bool()
    test_oct_int()
