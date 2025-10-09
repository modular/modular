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

from sys import size_of
from sys.info import CompilationTarget, is_64bit

from bit import count_leading_zeros
from builtin.simd import _modf
from testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from utils import StaticTuple
from utils.numerics import isfinite, isinf, isnan, nan


def test_cast():
    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, 1),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, -12),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    var b: UInt16 = 128
    assert_equal(Int(b.cast[DType.uint8]()), 128)
    assert_equal(Int(b.cast[DType.uint16]()), 128)
    assert_equal(Int(b.cast[DType.int8]()), -128)
    assert_equal(Int(b.cast[DType.int16]()), 128)

    assert_equal(
        BFloat16(33.0).cast[DType.float32]().cast[DType.bfloat16](), 33
    )
    assert_equal(Float16(33.0).cast[DType.float32]().cast[DType.float16](), 33)
    assert_equal(Float64(33.0).cast[DType.float32]().cast[DType.float16](), 33)

    # Test with a number right on the boundary of 32 bit and 64 bit, to make
    # sure the compiler can cast between the platform dependent types.
    alias u1 = Scalar[DType.uint](4294967296)
    alias i1 = Scalar[DType.int](4294967296)
    alias uc1 = i1.cast[DType.uint]()
    alias ic1 = u1.cast[DType.int]()
    assert_equal(uc1, u1)
    assert_equal(ic1, i1)

    @parameter
    if is_64bit():
        assert_equal(
            Scalar[DType.uint](18446744073709551615).cast[DType.int](),
            Scalar[DType.int](-1),
        )

        alias u2 = Scalar[DType.uint](18446744073709551615)
        alias i2 = Scalar[DType.int](-1)
        alias uc2 = i2.cast[DType.uint]()
        alias ic2 = u2.cast[DType.int]()
        assert_equal(uc2, u2)
        assert_equal(ic2, i2)
    else:
        assert_equal(
            Scalar[DType.uint](4294967295).cast[DType.int](),
            Scalar[DType.int](-1),
        )

        alias u3 = Scalar[DType.uint](4294967295)
        alias i3 = Scalar[DType.int](-1)
        alias uc3 = i3.cast[DType.uint]()
        alias ic3 = u3.cast[DType.int]()
        assert_equal(uc3, u3)
        assert_equal(ic3, i3)


def test_list_literal_ctor():
    var s: SIMD[DType.uint8, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
    assert_equal(s[0], 1)
    assert_equal(s[4], 5)
    assert_equal(s[7], 8)

    var s2: SIMD[DType.bool, 2] = [True, False]
    assert_true(s2[0])
    assert_false(s2[1])


def test_cast_init():
    # Basic casting preserves value within range
    assert_equal(Int8(UInt8(127)), Int8(127))

    # Numbers above signed max wrap to negative using two's complement
    assert_equal(Int8(UInt8(128)), Int8(-128))
    assert_equal(Int8(UInt8(129)), Int8(-127))
    assert_equal(Int8(UInt8(256)), Int8(0))

    # Negative signed convert to unsigned using two's complement
    assert_equal(UInt8(Int8(-128)), UInt8(128))
    assert_equal(UInt8(Int8(-127)), UInt8(129))
    assert_equal(UInt8(Int8(-1)), UInt8(255))

    # Truncate precision after downcast and upcast
    assert_equal(
        Float64(Float32(Float64(123456789.123456789))), Float64(123456792.0)
    )

    # Rightmost bits of significand become 0's on upcast
    assert_equal(Float64(Float32(0.3)), Float64(0.30000001192092896))

    # Numbers equal after truncation of float literal and cast truncation
    assert_equal(
        Float32(Float64(123456789.123456789)), Float32(123456789.123456789)
    )

    # Float to int/uint floors
    assert_equal(Int64(Float64(42.2)), Int64(42))

    # Pass a scalar to initialize a SIMD vector with more elements
    assert_equal(
        SIMD[DType.float64, 4](Float32(21.5)), SIMD[DType.float64, 4](21.5)
    )


def test_init_from_index():
    alias a = UInt.MAX
    alias a_str = String(a)
    assert_equal(a_str, String(UInt128(a)))
    assert_equal(a_str, String(Int128(a)))
    assert_equal(a_str, String(UInt256(a)))
    assert_equal(a_str, String(Int256(a)))


def test_from_bits():
    assert_true(Scalar[DType.bool](from_bits=UInt8(0x01)))
    assert_false(Scalar[DType.bool](from_bits=UInt8(0x00)))

    assert_equal(Int64(from_bits=UInt64(0xFFFFFFFFFFFFFFFF)), -1)
    assert_equal(UInt128(from_bits=Int128(-1)), -1)

    assert_equal(Float32(from_bits=UInt32(0x3F800000)), 1.0)
    assert_equal(Float32(from_bits=UInt32(0xBF800000)), -1.0)

    # Test from_bits with different integer types
    var uint32_bits = SIMD[DType.uint32, 4](
        0x3F800000, 0x40000000, 0x40400000, 0x40800000
    )
    var float32_from_bits = SIMD[DType.float32, 4](from_bits=uint32_bits)

    # These bit patterns represent 1.0, 2.0, 3.0, 4.0 in IEEE 754 float32
    assert_almost_equal(float32_from_bits[0], Float32(1.0), atol=1e-6)
    assert_almost_equal(float32_from_bits[1], Float32(2.0), atol=1e-6)
    assert_almost_equal(float32_from_bits[2], Float32(3.0), atol=1e-6)
    assert_almost_equal(float32_from_bits[3], Float32(4.0), atol=1e-6)

    # Test with int64 -> float64
    var uint64_bits = SIMD[DType.uint64, 2](
        0x3FF0000000000000, 0x4000000000000000
    )
    var float64_from_bits = SIMD[DType.float64, 2](from_bits=uint64_bits)

    assert_almost_equal(float64_from_bits[0], Float64(1.0), atol=1e-15)
    assert_almost_equal(float64_from_bits[1], Float64(2.0), atol=1e-15)

    # Test with int32 -> int32 (identity)
    var int32_bits = SIMD[DType.int32, 4](42, -42, 100, -100)
    var int32_from_bits = SIMD[DType.int32, 4](from_bits=int32_bits)
    assert_equal(int32_from_bits, int32_bits)


def test_to_bits():
    assert_equal(Scalar[DType.bool](True).to_bits(), 0x01)
    assert_equal(Scalar[DType.bool](False).to_bits(), 0x00)
    assert_equal(Scalar[DType.bool](True).to_bits[DType.uint8](), UInt8(0x01))

    assert_equal(Float32(1.0).to_bits(), 0x3F800000)
    assert_equal(Float32(-1.0).to_bits(), 0xBF800000)
    assert_equal(Float32(1.0).to_bits[DType.uint32](), UInt32(0x3F800000))
    assert_equal(Float32(1.0).to_bits[DType.uint64](), UInt64(0x3F800000))

    # Test to_bits conversion
    var float32_vals = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var bits = float32_vals.to_bits()

    # Convert back and check
    var reconstructed = SIMD[DType.float32, 4](from_bits=bits)
    assert_equal(reconstructed, float32_vals)

    # Test with different target bit width
    var int16_vals = SIMD[DType.int16, 4](1000, -1000, 2000, -2000)
    var uint16_bits = int16_vals.to_bits[DType.uint16]()

    # Should preserve bit patterns
    var reconstructed_int16 = SIMD[DType.int16, 4](from_bits=uint16_bits)
    assert_equal(reconstructed_int16, int16_vals)


def test_from_to_bits_roundtrip():
    alias dtypes = [
        DType.bool,
        DType.int,
        DType.uint,
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
        DType.uint64,
        DType.int64,
        DType.uint128,
        DType.int128,
        DType.uint256,
        DType.int256,
    ]

    @parameter
    for dt in dtypes:
        alias S = Scalar[dt]
        for n in range(-5, 5):
            var res = S(from_bits=S(n).to_bits())
            assert_equal(res, S(n))

    fn floating_point_dtypes() -> List[DType]:
        var res = [DType.float16, DType.float32, DType.float64]

        res.append(DType.bfloat16)
        return res^

    alias fp_dtypes = floating_point_dtypes()

    @parameter
    for dt in fp_dtypes:
        alias S = Scalar[dt]
        for i in range(-10, 10):
            var v = 1 / S(i)
            var res = S(from_bits=S(v).to_bits())
            assert_equal(res, S(v))


def test_simd_variadic():
    assert_equal(String(SIMD[DType.int, 4](52, 12, -43, 5)), "[52, 12, -43, 5]")
    assert_equal(String(SIMD[DType.uint, 4](52, 12, 43, 5)), "[52, 12, 43, 5]")


def test_convert_simd_to_string():
    var a: SIMD[DType.float32, 2] = 5
    assert_equal(String(a), "[5.0, 5.0]")

    var b: SIMD[DType.float64, 4] = 6
    assert_equal(String(b), "[6.0, 6.0, 6.0, 6.0]")

    var c: SIMD[DType.int, 8] = -7
    assert_equal(String(c), "[-7, -7, -7, -7, -7, -7, -7, -7]")

    var d: SIMD[DType.uint, 8] = 8
    assert_equal(String(d), "[8, 8, 8, 8, 8, 8, 8, 8]")

    # TODO: uncomment when https://github.com/modular/modular/issues/2353 is fixed
    # assert_equal(String(UInt32(-1)), "4294967295")
    assert_equal(String(UInt64(-1)), "18446744073709551615")

    assert_equal(String((UInt16(32768))), "32768")
    assert_equal(String((UInt16(65535))), "65535")
    assert_equal(String((Int16(-2))), "-2")

    assert_equal(String(UInt64(16646288086500911323)), "16646288086500911323")

    # https://github.com/modular/modular/issues/556
    assert_equal(
        String(
            SIMD[DType.uint64, 4](
                0xA0761D6478BD642F,
                0xE7037ED1A0B428DB,
                0x8EBC6AF09C88C6E3,
                0x589965CC75374CC3,
            )
        ),
        (
            "[11562461410679940143, 16646288086500911323, 10285213230658275043,"
            " 6384245875588680899]"
        ),
    )

    assert_equal(
        String(
            SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448)
        ),
        "[-943274556, -875902520, -808530484, -741158448]",
    )


def test_simd_repr():
    assert_equal(
        SIMD[DType.int32, 4](1, 2, 3, 4).__repr__(),
        "SIMD[DType.int32, 4](1, 2, 3, 4)",
    )
    assert_equal(
        SIMD[DType.int32, 4](-1, 2, -3, 4).__repr__(),
        "SIMD[DType.int32, 4](-1, 2, -3, 4)",
    )
    assert_equal(
        SIMD[DType.bool, 2](True, False).__repr__(),
        "SIMD[DType.bool, 2](True, False)",
    )
    assert_equal(Int32(4).__repr__(), "SIMD[DType.int32, 1](4)")
    assert_equal(
        Float64(235234523.3452).__repr__(),
        "SIMD[DType.float64, 1](235234523.3452)",
    )
    assert_equal(
        Float32(2897239).__repr__(), "SIMD[DType.float32, 1](2897239.0)"
    )
    assert_equal(Float16(324).__repr__(), "SIMD[DType.float16, 1](324.0)")
    assert_equal(
        SIMD[DType.float32, 4](
            Float32.MAX, Float32.MIN, -0.0, nan[DType.float32]()
        ).__repr__(),
        "SIMD[DType.float32, 4](inf, -inf, -0.0, nan)",
    )


def test_issue_1625():
    var size = 16
    alias simd_width = 8
    var ptr = UnsafePointer[Int64].alloc(size)
    for i in range(size):
        ptr[i] = i

    var x = ptr.load[width = 2 * simd_width](0)
    var evens_and_odds = x.deinterleave()

    # FIXME (40568) should directly use the SIMD assert_equal
    assert_equal(
        String(evens_and_odds[0]),
        String(SIMD[DType.int64, 8](0, 2, 4, 6, 8, 10, 12, 14)),
    )
    assert_equal(
        String(evens_and_odds[1]),
        String(SIMD[DType.int64, 8](1, 3, 5, 7, 9, 11, 13, 15)),
    )
    ptr.free()


def test_issue_20421():
    var a = UnsafePointer[UInt8].alloc(count=16 * 64, alignment=64)
    for i in range(16 * 64):
        a[i] = i & 255
    var av16 = (
        a.offset(128 + 64 + 4).bitcast[Int32]().load[width=4, alignment=1]()
    )
    assert_equal(
        av16,
        SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448),
    )
    a.free()


def test_issue_30237():
    alias dtype = DType.float32
    alias simd_width = 1
    alias coefficients_len = 7
    alias coefficients = InlineArray[SIMD[dtype, simd_width], coefficients_len](
        4.89352455891786e-03,
        6.37261928875436e-04,
        1.48572235717979e-05,
        5.12229709037114e-08,
        -8.60467152213735e-11,
        2.00018790482477e-13,
        -2.76076847742355e-16,
    )

    @parameter
    @always_inline
    fn eval1(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        var c_last = coefficients[coefficients_len - 1]
        var c_second_from_last = coefficients[coefficients_len - 2]

        var result = x.fma(c_last, c_second_from_last)

        @parameter
        for idx in range(coefficients_len - 2):
            var c = coefficients[coefficients_len - 3 - idx]
            result = x.fma(result, c)

        return result

    @parameter
    @always_inline
    fn eval2(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        var c_last = coefficients[coefficients_len - 1]
        var c_second_from_last = coefficients[coefficients_len - 2]

        var result = x.fma(c_last, c_second_from_last)

        for idx in range(coefficients_len - 2):
            var coefs = coefficients
            var c = coefs[coefficients_len - 3 - idx]
            result = x.fma(result, c)

        return result

    alias x = 6.0
    alias x2 = x * x
    var result1 = eval1(x2)
    var result2 = eval2(x2)

    assert_equal(result1, result2)


def test_bool():
    assert_true(Scalar[DType.bool](True).__bool__())
    assert_false(Scalar[DType.bool](False).__bool__())
    assert_true(Int32(5).__bool__())
    assert_false(Int32(0).__bool__())
    assert_true(Float32(5.0).__bool__())
    assert_false(Float32(0.0).__bool__())


def test_truthy():
    alias dtypes = (
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float16,
        DType.float32,
        DType.float64,
        DType.int,
        DType.uint,
    )

    @parameter
    fn test_dtype[dtype: DType]() raises:
        # Scalars of 0-values are false-y, 1-values are truth-y
        assert_false(Scalar[dtype](0))
        assert_true(Scalar[dtype](1))

    @parameter
    for i in range(dtypes.__len__()):
        alias dtype = dtypes[i]
        test_dtype[dtype]()

    test_dtype[DType.bfloat16]()


def test_len():
    var i1 = Int32(0)
    assert_equal(i1.__len__(), 1)

    alias I32 = SIMD[DType.int32, 4]
    var i2 = I32(-1)
    assert_equal(4, i2.__len__())
    var i3 = I32(-1, 0, 1, 3)
    assert_equal(4, i3.__len__())

    var i4 = Int8(1)
    assert_equal(1, i4.__len__())

    alias UI64 = SIMD[DType.uint64, 16]
    var i5 = UI64(10)
    assert_equal(16, i5.__len__())
    var i6 = UI64(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert_equal(16, i6.__len__())

    alias BF16 = SIMD[DType.bfloat16, 2]
    var f1 = BF16(0.0)
    assert_equal(2, f1.__len__())
    var f2 = BF16(0.1, 0.2)
    assert_equal(2, f2.__len__())

    alias F = SIMD[DType.float64, 8]
    var f3 = F(1.0)
    assert_equal(8, f3.__len__())
    var f4 = F(0, -1.0, 1.0, -1.111, 1.111, -2.2222, 2.2222, 3.1415)
    assert_equal(8, f4.__len__())


def test_add():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__add__(0), I(-2, -4, 0, 1))
    assert_equal(i.__add__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__add__(2), I(0, -2, 2, 3))
    assert_equal(i.__add__(Int32(2)), I(0, -2, 2, 3))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__add__(i2), I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__add__(f2), F(0, 0, 0, 0, 0, 0, 0, 0))


def test_radd():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__radd__(0), I(-2, -4, 0, 1))
    assert_equal(i.__radd__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__radd__(2), I(0, -2, 2, 3))
    assert_equal(i.__radd__(Int32(2)), I(0, -2, 2, 3))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__radd__(i2), I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__radd__(f2), F(0, 0, 0, 0, 0, 0, 0, 0))


def test_iadd():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    i.__iadd__(0)
    assert_equal(i, I(-2, -4, 0, 1))
    i.__iadd__(Int32(0))
    assert_equal(i, I(-2, -4, 0, 1))
    i.__iadd__(2)
    assert_equal(i, I(0, -2, 2, 3))
    i.__iadd__(I(0, -2, 2, 3))
    assert_equal(i, I(0, -4, 4, 6))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    i1.__iadd__(i2)
    assert_equal(i1, I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    f1.__iadd__(f2)
    assert_equal(f1, F(0, 0, 0, 0, 0, 0, 0, 0))


def test_sub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__sub__(0), I(-2, -4, 0, 1))
    assert_equal(i.__sub__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__sub__(2), I(-4, -6, -2, -1))
    assert_equal(i.__sub__(Int32(2)), I(-4, -6, -2, -1))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__sub__(i2), I(-1, -9, -6, 1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__sub__(f2), F(2, -2, 2, -2, 2, -2, 2, -2))


def test_rsub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__rsub__(0), I(2, 4, 0, -1))
    assert_equal(i.__rsub__(Int32(0)), I(2, 4, 0, -1))
    assert_equal(i.__rsub__(2), I(4, 6, 2, 1))
    assert_equal(i.__rsub__(Int32(2)), I(4, 6, 2, 1))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__rsub__(i2), I(1, 9, 6, -1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__rsub__(f2), F(-2, 2, -2, 2, -2, 2, -2, 2))


def test_isub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    i.__isub__(0)
    assert_equal(i, I(-2, -4, 0, 1))
    i.__isub__(Int32(0))
    assert_equal(i, I(-2, -4, 0, 1))
    i.__isub__(2)
    assert_equal(i, I(-4, -6, -2, -1))
    i.__isub__(I(0, -2, 2, 3))
    assert_equal(i, I(-4, -4, -4, -4))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    i1.__isub__(i2)
    assert_equal(i1, I(-1, -9, -6, 1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    f1.__isub__(f2)
    assert_equal(f1, F(2, -2, 2, -2, 2, -2, 2, -2))


def test_ceil():
    assert_equal(Float32.__ceil__(Float32(1.5)), 2.0)
    assert_equal(Float32.__ceil__(Float32(-1.5)), -1.0)
    assert_equal(Float32.__ceil__(Float32(3.0)), 3.0)

    assert_equal(BFloat16.__ceil__(BFloat16(2.5)), 3.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__ceil__(F(0.0, 1.4, -42.5, -12.6)), F(0.0, 2.0, -42.0, -12.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__ceil__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__ceil__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__ceil__(b), b)


def test_floor():
    assert_equal(Float32.__floor__(Float32(1.5)), 1.0)
    assert_equal(Float32.__floor__(Float32(-1.5)), -2.0)
    assert_equal(Float32.__floor__(Float32(3.0)), 3.0)

    assert_equal(BFloat16.__floor__(BFloat16(2.5)), 2.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__floor__(F(0.0, 1.6, -42.5, -12.4)), F(0.0, 1.0, -43.0, -13.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__floor__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__floor__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__floor__(b), b)


def test_trunc():
    assert_equal(Float32.__trunc__(Float32(1.5)), 1.0)
    assert_equal(Float32.__trunc__(Float32(-1.5)), -1.0)
    assert_equal(Float32.__trunc__(Float32(3.0)), 3.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__trunc__(F(0.0, 1.6, -42.5, -12.4)), F(0.0, 1.0, -42.0, -12.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__trunc__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__trunc__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__trunc__(b), b)


def test_round():
    assert_equal(Float32.__round__(Float32(2.5)), 2.0)
    assert_equal(Float32.__round__(Float32(3.5)), 4.0)
    assert_equal(Float32.__round__(Float32(-3.5)), -4.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(F.__round__(F(1.5, 2.5, -2.5, -3.5)), F(2.0, 2.0, -2.0, -4.0))


def test_div():
    assert_false(isfinite(Float32(33).__truediv__(0)))
    assert_false(isfinite(Float32(0).__truediv__(0)))

    assert_true(isinf(Float32(33).__truediv__(0)))
    assert_false(isinf(Float32(0).__truediv__(0)))

    assert_false(isnan(Float32(33).__truediv__(0)))
    assert_true(isnan(Float32(0).__truediv__(0)))

    alias F32 = SIMD[DType.float32, 4]
    var res = F32.__truediv__(F32(1, 0, 3, -1), F32(0, 0, 1, 0))
    alias B = SIMD[DType.bool, 4]
    assert_equal(isfinite(res), B(False, False, True, False))
    assert_equal(isinf(res), B(True, False, False, True))
    assert_equal(isnan(res), B(False, True, False, False))


def test_floordiv():
    assert_equal(Int32(2).__floordiv__(2), 1)
    assert_equal(Int32(2).__floordiv__(Int32(2)), 1)
    assert_equal(Int32(2).__floordiv__(Int32(3)), 0)

    assert_equal(Int32(2).__floordiv__(-2), -1)
    assert_equal(Int32(2).__floordiv__(Int32(-2)), -1)
    assert_equal(Int32(99).__floordiv__(Int32(-2)), -50)

    assert_equal(UInt32(2).__floordiv__(2), 1)
    assert_equal(UInt32(2).__floordiv__(UInt32(2)), 1)
    assert_equal(UInt32(2).__floordiv__(UInt32(3)), 0)

    assert_equal(Float32(2).__floordiv__(2), 1)
    assert_equal(Float32(2).__floordiv__(Float32(2)), 1)
    assert_equal(Float32(2).__floordiv__(Float32(3)), 0)

    assert_equal(Float32(2).__floordiv__(-2), -1)
    assert_equal(Float32(2).__floordiv__(Float32(-2)), -1)
    assert_equal(Float32(99).__floordiv__(Float32(-2)), -50)

    alias I = SIMD[DType.int32, 4]
    var i = I(2, 4, -2, -4)
    assert_equal(i.__floordiv__(2), I(1, 2, -1, -2))
    assert_equal(i.__floordiv__(Int32(2)), I(1, 2, -1, -2))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__floordiv__(3), F(1, -2, 0, 1))
    assert_equal(f.__floordiv__(Float32(3)), F(1, -2, 0, 1))


def test_rfloordiv():
    alias I = SIMD[DType.int32, 4]
    var i = I(2, 4, -2, -4)
    assert_equal(i.__rfloordiv__(2), I(1, 0, -1, -1))
    assert_equal(i.__rfloordiv__(Int32(2)), I(1, 0, -1, -1))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__rfloordiv__(3), F(1, -1, 3, 0))
    assert_equal(f.__rfloordiv__(Float32(3)), F(1, -1, 3, 0))


def test_mod():
    assert_equal(Int32(99) % Int32(1), 0)
    assert_equal(Int32(99) % Int32(3), 0)
    assert_equal(Int32(99) % Int32(-2), -1)
    assert_equal(Int32(99) % Int32(8), 3)
    assert_equal(Int32(99) % Int32(-8), -5)
    assert_equal(Int32(2) % Int32(-1), 0)
    assert_equal(Int32(2) % Int32(-2), 0)

    assert_equal(UInt32(99) % UInt32(1), 0)
    assert_equal(UInt32(99) % UInt32(3), 0)

    assert_equal(
        SIMD[DType.int32, 2](7, 7) % Int(4), SIMD[DType.int32, 2](3, 3)
    )

    var a = SIMD[DType.float32, 16](
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
    )
    var b = SIMD[DType.float32, 16](
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.1,
        -3.1,
        3.1,
        -3.1,
    )
    assert_equal(
        a % b,
        SIMD[DType.float32, 16](
            3.0999999046325684,
            0.89999985694885254,
            0.69999980926513672,
            -0.10000014305114746,
            -1.3000001907348633,
            -0.5000002384185791,
            0.10000014305114746,
            1.3000001907348633,
            0.5000002384185791,
            -3.0999999046325684,
            -0.89999985694885254,
            -0.69999980926513672,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )


def test_rmod():
    assert_equal(Int32(3).__rmod__(Int(4)), 1)

    alias I = SIMD[DType.int32, 2]
    var i = I(78, 78)
    assert_equal(i.__rmod__(Int(78)), I(0, 0))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__rmod__(3), F(0, -1, 0, 3))
    assert_equal(f.__rmod__(Float32(3)), F(0, -1, 0, 3))


def test_rotate():
    # Test with larger vectors and different data types
    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[5](),
        SIMD[DType.uint32, 8](1, 0, 1, 0, 0, 1, 0, 1),
    )

    # Test systematic rotation with 4-element vector
    alias simd_width = 4
    alias type = DType.uint32
    var base_pattern = SIMD[type, simd_width](1, 0, 1, 1)

    # Test rotate_left with all positions and negative values
    assert_equal(
        base_pattern.rotate_left[0](), SIMD[type, simd_width](1, 0, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_left[1](), SIMD[type, simd_width](0, 1, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_left[2](), SIMD[type, simd_width](1, 1, 1, 0)
    )
    assert_equal(
        base_pattern.rotate_left[3](), SIMD[type, simd_width](1, 1, 0, 1)
    )
    assert_equal(
        base_pattern.rotate_left[-1](), SIMD[type, simd_width](1, 1, 0, 1)
    )
    assert_equal(
        base_pattern.rotate_left[-2](), SIMD[type, simd_width](1, 1, 1, 0)
    )
    assert_equal(
        base_pattern.rotate_left[-3](), SIMD[type, simd_width](0, 1, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_left[-4](), SIMD[type, simd_width](1, 0, 1, 1)
    )

    # Test rotate_right with all positions and negative values
    assert_equal(
        base_pattern.rotate_right[0](), SIMD[type, simd_width](1, 0, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_right[1](), SIMD[type, simd_width](1, 1, 0, 1)
    )
    assert_equal(
        base_pattern.rotate_right[2](), SIMD[type, simd_width](1, 1, 1, 0)
    )
    assert_equal(
        base_pattern.rotate_right[3](), SIMD[type, simd_width](0, 1, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_right[4](), SIMD[type, simd_width](1, 0, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_right[-1](), SIMD[type, simd_width](0, 1, 1, 1)
    )
    assert_equal(
        base_pattern.rotate_right[-2](), SIMD[type, simd_width](1, 1, 1, 0)
    )
    assert_equal(
        base_pattern.rotate_right[-3](), SIMD[type, simd_width](1, 1, 0, 1)
    )

    # Test with sequential patterns for easier verification
    var sequential = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(sequential.rotate_left[1](), SIMD[DType.int32, 4](2, 3, 4, 1))
    assert_equal(sequential.rotate_left[2](), SIMD[DType.int32, 4](3, 4, 1, 2))
    assert_equal(sequential.rotate_right[1](), SIMD[DType.int32, 4](4, 1, 2, 3))
    assert_equal(sequential.rotate_right[2](), SIMD[DType.int32, 4](3, 4, 1, 2))

    # Test with 8-element vector
    var simd8 = SIMD[DType.uint8, 8](0, 1, 2, 3, 4, 5, 6, 7)
    var rotate8_3 = simd8.rotate_left[3]()
    var expected8_3 = SIMD[DType.uint8, 8](3, 4, 5, 6, 7, 0, 1, 2)
    assert_equal(rotate8_3, expected8_3)


def test_shift():
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).shift_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](11, 0, 13, 12, 0, 100, 0, 0).shift_right[5](),
        SIMD[DType.uint32, 8](0, 0, 0, 0, 0, 11, 0, 13),
    )

    assert_equal(
        SIMD[DType.float64, 8](11.1, 0, 13.1, 12.2, 0, 100.4, 0, 0).shift_right[
            5
        ](),
        SIMD[DType.float64, 8](0, 0, 0, 0, 0, 11.1, 0, 13.1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[1](),
        SIMD[type, simd_width](0, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[2](),
        SIMD[type, simd_width](1, 1, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[3](),
        SIMD[type, simd_width](1, 0, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[1](),
        SIMD[type, simd_width](0, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[2](),
        SIMD[type, simd_width](0, 0, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[3](),
        SIMD[type, simd_width](0, 0, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )


def test_shuffle():
    alias dtype = DType.int32
    alias width = 4

    vec = SIMD[dtype, width](100, 101, 102, 103)

    assert_equal(
        vec.shuffle[3, 2, 1, 0](), SIMD[dtype, width](103, 102, 101, 100)
    )
    assert_equal(
        vec.shuffle[0, 2, 4, 6](vec), SIMD[dtype, width](100, 102, 100, 102)
    )

    assert_equal(
        vec._shuffle_variadic[7, 6, 5, 4, 3, 2, 1, 0, output_size = 2 * width](
            vec
        ),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )

    assert_equal(
        vec._shuffle_list[width, StaticTuple[Int, width](3, 2, 1, 0)](vec),
        SIMD[dtype, width](103, 102, 101, 100),
    )
    assert_equal(
        vec._shuffle_list[width, StaticTuple[Int, width](0, 2, 4, 6)](vec),
        SIMD[dtype, width](100, 102, 100, 102),
    )

    assert_equal(
        vec._shuffle_list[
            2 * width, StaticTuple[Int, 2 * width](7, 6, 5, 4, 3, 2, 1, 0)
        ](vec),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )


def test_shuffle_dynamic_size_4_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    indices = SIMD[DType.uint8, 4](3, 3, 5, 5)

    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 4](30, 30, 50, 50)
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_8_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    # Let's use size 8
    indices = SIMD[DType.uint8, 8](3, 3, 5, 5, 7, 7, 9, 0)

    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 8](30, 30, 50, 50, 70, 70, 90, 0)
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_16_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    var indices = SIMD[DType.uint8, 16](
        3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 0, 1
    )
    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 16](
        30, 30, 50, 50, 70, 70, 90, 90, 110, 110, 130, 130, 150, 150, 0, 10
    )
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_32_uint8():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    # fmt: off
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices)

    expected_result = SIMD[DType.uint8, 32](
        30 , 30 , 50 , 50 , 70 , 70 , 90 , 90 ,
        110, 110, 130, 130, 150, 150, 0  , 10 ,
        0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 ,
        80 , 90 , 100, 110, 120, 130, 140, 150,
    )
    # fmt: on
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_64_uint8():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    # fmt: off
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices.join(indices))

    expected_result = SIMD[DType.uint8, 32](
        30 , 30 , 50 , 50 , 70 , 70 , 90 , 90 ,
        110, 110, 130, 130, 150, 150, 0  , 10 ,
        0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 ,
        80 , 90 , 100, 110, 120, 130, 140, 150,
    )
    # fmt: on
    assert_equal(result, expected_result.join(expected_result))


def test_shuffle_dynamic_size_32_float():
    # fmt: off
    var table_lookup = SIMD[DType.float64, 16](
        0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
    )
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices)

    expected_result = SIMD[DType.float64, 32](
        30. , 30. , 50. , 50. , 70. , 70. , 90. , 90. ,
        110., 110., 130., 130., 150., 150., 0.  , 10. ,
        0.  , 10. , 20. , 30. , 40. , 50. , 60. , 70. ,
        80. , 90. , 100., 110., 120., 130., 140., 150.,
    )
    # fmt: on
    assert_equal(result, expected_result)


def test_insert():
    assert_equal(Int32(3).insert(Int32(4)), 4)

    assert_equal(
        SIMD[DType.int, 4](0, 1, 2, 3).insert(SIMD[DType.int, 2](9, 6)),
        SIMD[DType.int, 4](9, 6, 2, 3),
    )

    assert_equal(
        SIMD[DType.int, 4](0, 1, 2, 3).insert[offset=2](
            SIMD[DType.int, 2](9, 6)
        ),
        SIMD[DType.int, 4](0, 1, 9, 6),
    )

    assert_equal(
        SIMD[DType.uint, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=4](
            SIMD[DType.uint, 4](9, 6, 3, 7)
        ),
        SIMD[DType.uint, 8](0, 1, 2, 3, 9, 6, 3, 7),
    )

    assert_equal(
        SIMD[DType.uint, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=0](
            SIMD[DType.uint, 4](9, 6, 3, 7)
        ),
        SIMD[DType.uint, 8](9, 6, 3, 7, 5, 6, 7, 8),
    )


def test_join():
    alias I2 = SIMD[DType.int32, 2]
    assert_equal(Int32(3).join(Int32(4)), I2(3, 4))

    alias I4 = SIMD[DType.int32, 4]
    assert_equal(I2(5, 6).join(I2(9, 10)), I4(5, 6, 9, 10))

    vec = I4(100, 101, 102, 103)
    assert_equal(
        vec.join(vec),
        SIMD[DType.int32, 8](100, 101, 102, 103, 100, 101, 102, 103),
    )


def test_interleave():
    assert_equal(
        String(Int32(0).interleave(Int32(1))),
        String(SIMD[DType.int32, 2](0, 1)),
    )

    assert_equal(
        SIMD[DType.int, 2](0, -2).interleave(SIMD[DType.int, 2](1, -3)),
        SIMD[DType.int, 4](0, 1, -2, -3),
    )

    assert_equal(
        SIMD[DType.uint, 2](0, 2).interleave(SIMD[DType.uint, 2](1, 3)),
        SIMD[DType.uint, 4](0, 1, 2, 3),
    )


def test_deinterleave():
    var tup2 = SIMD[DType.float32, 2](1, 2).deinterleave()
    assert_equal(tup2[0], Float32(1))
    assert_equal(tup2[1], Float32(2))

    var tup4 = SIMD[DType.int, 4](0, 1, -2, -3).deinterleave()
    assert_equal(tup4[0], __type_of(tup4[0])(0, -2))
    assert_equal(tup4[1], __type_of(tup4[0])(1, -3))

    var tup8 = SIMD[DType.uint, 8](0, 1, 2, 3, 4, 5, 6, 7).deinterleave()
    assert_equal(tup8[0], __type_of(tup8[0])(0, 2, 4, 6))
    assert_equal(tup8[1], __type_of(tup8[0])(1, 3, 5, 7))


def test_extract():
    alias s1 = Int64(99).slice[1]()  # test compile time
    alias s2 = Int64(99).slice[1, offset=0]()
    assert_equal(s1, 99)
    assert_equal(s2, 99)

    assert_equal(
        SIMD[DType.int, 4](99, 1, 2, 4).slice[4](),
        SIMD[DType.int, 4](99, 1, 2, 4),
    )

    assert_equal(
        SIMD[DType.int, 4](99, 1, 2, 4).slice[2, offset=0](),
        SIMD[DType.int, 2](99, 1),
    )

    assert_equal(
        SIMD[DType.uint, 4](99, 1, 2, 4).slice[2, offset=2](),
        SIMD[DType.uint, 2](2, 4),
    )

    assert_equal(
        SIMD[DType.uint, 4](99, 1, 2, 4).slice[2, offset=1](),
        SIMD[DType.uint, 2](1, 2),
    )


def test_limits():
    @parameter
    fn test_integral_overflow[dtype: DType]() raises:
        var max_value = Scalar[dtype].MAX
        var min_value = Scalar[dtype].MIN
        assert_equal(max_value + 1, min_value)

    test_integral_overflow[DType.int]()
    test_integral_overflow[DType.uint]()
    test_integral_overflow[DType.int8]()
    test_integral_overflow[DType.uint8]()
    test_integral_overflow[DType.int16]()
    test_integral_overflow[DType.uint16]()
    test_integral_overflow[DType.int32]()
    test_integral_overflow[DType.uint32]()
    test_integral_overflow[DType.int64]()
    test_integral_overflow[DType.uint64]()


def test_abs():
    assert_equal(abs(Float32(1.0)), 1)
    assert_equal(abs(Float32(-1.0)), 1)
    assert_equal(abs(Float32(0.0)), 0)
    assert_equal(
        abs(SIMD[DType.float32, 4](0.0, 1.5, -42.5, -12.7)),
        SIMD[DType.float32, 4](0.0, 1.5, 42.5, 12.7),
    )
    assert_equal(
        abs(SIMD[DType.int32, 4](0, 2, -42, -12)),
        SIMD[DType.int32, 4](0, 2, 42, 12),
    )
    assert_equal(
        abs(SIMD[DType.uint32, 4](0, 2, 42, 12)),
        SIMD[DType.uint32, 4](0, 2, 42, 12),
    )
    assert_equal(
        abs(SIMD[DType.bool, 4](True, False, True, False)),
        SIMD[DType.bool, 4](True, False, True, False),
    )


def test_clamp():
    # Basic clamp tests
    alias F = SIMD[DType.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(f.clamp(-6.0, 5.5), F(-6.0, -5.0, 5.0, 5.5))

    alias I = SIMD[DType.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(i.clamp(-7, 4), I(-7, -5, 4, 4))

    # Test clamping with edge cases
    var simd = SIMD[DType.float32, 8](
        -10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0, 20.0
    )
    var lower = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var upper = SIMD[DType.float32, 8](
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0
    )

    var clamped = simd.clamp(lower, upper)
    var expected = SIMD[DType.float32, 8](
        0.0, 0.0, 0.0, 1.0, 5.0, 10.0, 10.0, 10.0
    )

    assert_equal(clamped, expected)

    # Test with integer types
    var int_simd = SIMD[DType.int32, 4](-5, 0, 10, 20)
    var int_lower = SIMD[DType.int32, 4](0, 0, 0, 0)
    var int_upper = SIMD[DType.int32, 4](15, 15, 15, 15)

    var int_clamped = int_simd.clamp(int_lower, int_upper)
    var int_expected = SIMD[DType.int32, 4](0, 0, 10, 15)

    assert_equal(int_clamped, int_expected)

    # Test where lower == upper (should clamp to that value)
    var constant_bounds = SIMD[DType.int32, 4](5, 5, 5, 5)
    var const_clamped = int_simd.clamp(constant_bounds, constant_bounds)
    var const_expected = SIMD[DType.int32, 4](5, 5, 5, 5)

    assert_equal(const_clamped, const_expected)


def test_indexing():
    var s = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(s[False], 1)
    assert_equal(s[Int(2)], 3)
    assert_equal(s[3], 4)


def test_reduce():
    @parameter
    def test_dtype[dtype: DType]():
        alias X8 = SIMD[dtype, 8]
        alias X4 = SIMD[dtype, 4]
        alias X2 = SIMD[dtype, 2]
        alias X1 = Scalar[dtype]
        var x8: X8
        var x4: X4
        var x2: X2
        var x1: X1

        @parameter
        if dtype.is_numeric():
            # reduce_add
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 6, 8, 10)
            x2 = X2(12, 16)
            x1 = X1(Int(28))  # TODO: fix MOCO-697 and use X1(28) instead
            assert_equal(x8.reduce_add(), x1)
            assert_equal(x4.reduce_add(), x1)
            assert_equal(x2.reduce_add(), x1)
            assert_equal(x1.reduce_add(), x1)
            assert_equal(x8.reduce_add[2](), x2)
            assert_equal(x4.reduce_add[2](), x2)
            assert_equal(x2.reduce_add[2](), x2)
            assert_equal(x8.reduce_add[4](), x4)
            assert_equal(x4.reduce_add[4](), x4)
            assert_equal(x8.reduce_add[8](), x8)
            assert_equal(X2(6, 3).reduce_add(), 9)

            # reduce_mul
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 5, 12, 21)
            x2 = X2(0, 105)
            x1 = X1(Int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_mul(), x1)
            assert_equal(x4.reduce_mul(), x1)
            assert_equal(x2.reduce_mul(), x1)
            assert_equal(x1.reduce_mul(), x1)
            assert_equal(x8.reduce_mul[2](), x2)
            assert_equal(x4.reduce_mul[2](), x2)
            assert_equal(x2.reduce_mul[2](), x2)
            assert_equal(x8.reduce_mul[4](), x4)
            assert_equal(x4.reduce_mul[4](), x4)
            assert_equal(x8.reduce_mul[8](), x8)
            assert_equal(X2(6, 3).reduce_mul(), 18)

            # reduce_min
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 1, 2, 3)
            x2 = X2(0, 1)
            x1 = X1(Int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_min(), x1)
            assert_equal(x4.reduce_min(), x1)
            assert_equal(x2.reduce_min(), x1)
            assert_equal(x1.reduce_min(), x1)
            assert_equal(x8.reduce_min[2](), x2)
            assert_equal(x4.reduce_min[2](), x2)
            assert_equal(x2.reduce_min[2](), x2)
            assert_equal(x8.reduce_min[4](), x4)
            assert_equal(x4.reduce_min[4](), x4)
            assert_equal(x8.reduce_min[8](), x8)
            assert_equal(X2(6, 3).reduce_min(), 3)

            # reduce_max
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 5, 6, 7)
            x2 = X2(6, 7)
            x1 = X1(Int(7))  # TODO: fix MOCO-697 and use X1(7) instead
            assert_equal(x8.reduce_max(), x1)
            assert_equal(x4.reduce_max(), x1)
            assert_equal(x2.reduce_max(), x1)
            assert_equal(x1.reduce_max(), x1)
            assert_equal(x8.reduce_max[2](), x2)
            assert_equal(x4.reduce_max[2](), x2)
            assert_equal(x2.reduce_max[2](), x2)
            assert_equal(x8.reduce_max[4](), x4)
            assert_equal(x4.reduce_max[4](), x4)
            assert_equal(x8.reduce_max[8](), x8)
            assert_equal(X2(6, 3).reduce_max(), 6)

        @parameter
        if dtype.is_signed():
            # reduce_add
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(4, -6, 8, -10)
            x2 = X2(12, -16)
            x1 = X1(Int(-4))  # TODO: fix MOCO-697 and use X1(-4) instead
            assert_equal(x8.reduce_add(), x1)
            assert_equal(x4.reduce_add(), x1)
            assert_equal(x2.reduce_add(), x1)
            assert_equal(x1.reduce_add(), x1)
            assert_equal(x8.reduce_add[2](), x2)
            assert_equal(x4.reduce_add[2](), x2)
            assert_equal(x2.reduce_add[2](), x2)
            assert_equal(x8.reduce_add[4](), x4)
            assert_equal(x4.reduce_add[4](), x4)
            assert_equal(x8.reduce_add[8](), x8)
            assert_equal(X2(6, -3).reduce_add(), 3)

            # reduce_mul
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(0, 5, 12, 21)
            x2 = X2(0, 105)
            x1 = X1(Int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_mul(), x1)
            assert_equal(x4.reduce_mul(), x1)
            assert_equal(x2.reduce_mul(), x1)
            assert_equal(x1.reduce_mul(), x1)
            assert_equal(x8.reduce_mul[2](), x2)
            assert_equal(x4.reduce_mul[2](), x2)
            assert_equal(x2.reduce_mul[2](), x2)
            assert_equal(x8.reduce_mul[4](), x4)
            assert_equal(x4.reduce_mul[4](), x4)
            assert_equal(x8.reduce_mul[8](), x8)
            assert_equal(X2(6, -3).reduce_mul(), -18)

            # reduce_min
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(0, -5, 2, -7)
            x2 = X2(0, -7)
            x1 = X1(Int(-7))  # TODO: fix MOCO-697 and use X1(-7) instead
            assert_equal(x8.reduce_min(), x1)
            assert_equal(x4.reduce_min(), x1)
            assert_equal(x2.reduce_min(), x1)
            assert_equal(x1.reduce_min(), x1)
            assert_equal(x8.reduce_min[2](), x2)
            assert_equal(x4.reduce_min[2](), x2)
            assert_equal(x2.reduce_min[2](), x2)
            assert_equal(x8.reduce_min[4](), x4)
            assert_equal(x4.reduce_min[4](), x4)
            assert_equal(x8.reduce_min[8](), x8)
            assert_equal(X2(6, -3).reduce_min(), -3)

            # reduce_max
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(4, -1, 6, -3)
            x2 = X2(6, -1)
            x1 = X1(Int(6))  # TODO: fix MOCO-697 and use X1(6) instead
            assert_equal(x8.reduce_max(), x1)
            assert_equal(x4.reduce_max(), x1)
            assert_equal(x2.reduce_max(), x1)
            assert_equal(x1.reduce_max(), x1)
            assert_equal(x8.reduce_max[2](), x2)
            assert_equal(x4.reduce_max[2](), x2)
            assert_equal(x2.reduce_max[2](), x2)
            assert_equal(x8.reduce_max[4](), x4)
            assert_equal(x4.reduce_max[4](), x4)
            assert_equal(x8.reduce_max[8](), x8)
            assert_equal(X2(6, -3).reduce_max(), 6)

        @parameter
        if dtype is DType.bool:
            # reduce_and
            var x8b = SIMD[DType.bool, 8](
                False, False, True, True, False, True, False, True
            )
            var x4b = SIMD[DType.bool, 4](False, False, False, True)
            var x2b = SIMD[DType.bool, 2](False, False)
            var x1b = Scalar[DType.bool](False)
            assert_equal(x8b.reduce_and(), x1b)
            assert_equal(x4b.reduce_and(), x1b)
            assert_equal(x2b.reduce_and(), x1b)
            assert_equal(x1b.reduce_and(), x1b)
            assert_equal(x8b.reduce_and[2](), x2b)
            assert_equal(x4b.reduce_and[2](), x2b)
            assert_equal(x2b.reduce_and[2](), x2b)
            assert_equal(x8b.reduce_and[4](), x4b)
            assert_equal(x4b.reduce_and[4](), x4b)
            assert_equal(x8b.reduce_and[8](), x8b)
            assert_equal(SIMD[DType.bool, 2](True, True).reduce_and(), True)

            # reduce_or
            x8b = SIMD[DType.bool, 8](
                False, False, True, True, False, True, False, True
            )
            x4b = SIMD[DType.bool, 4](False, True, True, True)
            x2b = SIMD[DType.bool, 2](True, True)
            x1b = Scalar[DType.bool](True)
            assert_equal(x8b.reduce_or(), x1b)
            assert_equal(x4b.reduce_or(), x1b)
            assert_equal(x2b.reduce_or(), x1b)
            assert_equal(x1b.reduce_or(), x1b)
            assert_equal(x8b.reduce_or[2](), x2b)
            assert_equal(x4b.reduce_or[2](), x2b)
            assert_equal(x2b.reduce_or[2](), x2b)
            assert_equal(x8b.reduce_or[4](), x4b)
            assert_equal(x4b.reduce_or[4](), x4b)
            assert_equal(x8b.reduce_or[8](), x8b)
            assert_equal(SIMD[DType.bool, 2](False, False).reduce_or(), False)

        @parameter
        if dtype.is_integral():
            # reduce_and
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 1, 2, 3)
            x2 = X2(0, 1)
            x1 = X1(Int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_and(), x1)
            assert_equal(x4.reduce_and(), x1)
            assert_equal(x2.reduce_and(), x1)
            assert_equal(x1.reduce_and(), x1)
            assert_equal(x8.reduce_and[2](), x2)
            assert_equal(x4.reduce_and[2](), x2)
            assert_equal(x2.reduce_and[2](), x2)
            assert_equal(x8.reduce_and[4](), x4)
            assert_equal(x4.reduce_and[4](), x4)
            assert_equal(x8.reduce_and[8](), x8)
            assert_equal(X2(6, 3).reduce_and(), 2)

            # reduce_or
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 5, 6, 7)
            x2 = X2(6, 7)
            x1 = X1(Int(7))  # TODO: fix MOCO-697 and use X1(7) instead
            assert_equal(x8.reduce_or(), x1)
            assert_equal(x4.reduce_or(), x1)
            assert_equal(x2.reduce_or(), x1)
            assert_equal(x1.reduce_or(), x1)
            assert_equal(x8.reduce_or[2](), x2)
            assert_equal(x4.reduce_or[2](), x2)
            assert_equal(x2.reduce_or[2](), x2)
            assert_equal(x8.reduce_or[4](), x4)
            assert_equal(x4.reduce_or[4](), x4)
            assert_equal(x8.reduce_or[8](), x8)
            assert_equal(X2(6, 3).reduce_or(), 7)

    test_dtype[DType.bool]()
    test_dtype[DType.int8]()
    test_dtype[DType.int16]()
    test_dtype[DType.int32]()
    test_dtype[DType.int64]()
    test_dtype[DType.uint8]()
    test_dtype[DType.uint16]()
    test_dtype[DType.uint32]()
    test_dtype[DType.uint64]()
    test_dtype[DType.float16]()
    test_dtype[DType.float32]()
    test_dtype[DType.float64]()
    test_dtype[DType.int]()
    test_dtype[DType.uint]()

    test_dtype[DType.bfloat16]()


def test_reduce_bit_count():
    var int_0xFFFF = Int32(0xFFFF)
    assert_equal(int_0xFFFF.reduce_bit_count(), 16)

    var int_iota8 = SIMD[DType.int32, 8](0, 1, 2, 3, 4, 5, 6, 7)
    assert_equal(int_iota8.reduce_bit_count(), 12)

    var bool_true = Scalar[DType.bool](True)
    assert_equal(bool_true.reduce_bit_count(), 1)

    var bool_false = Scalar[DType.bool](False)
    assert_equal(bool_false.reduce_bit_count(), 0)

    var bool_true16 = SIMD[DType.bool, 16](fill=True)
    assert_equal(bool_true16.reduce_bit_count(), 16)


def test_pow():
    alias nan = FloatLiteral.nan
    alias neg_zero = FloatLiteral.negative_zero
    alias inf = FloatLiteral.infinity
    alias neg_inf = FloatLiteral.negative_infinity

    # Float32 tests
    alias F32x4 = SIMD[DType.float32, 4]
    alias F32x8 = SIMD[DType.float32, 8]

    var f32x4_val = F32x4(0, 1, 2, 3)
    var f32x8_val = F32x8(0, 1, 2, 3, 4, 5, 6, 7)
    assert_equal(f32x4_val.__pow__(10.0), F32x4(0.0, 1.0, 1024.0, 59049.0))
    assert_almost_equal(
        f32x8_val.__pow__(15.0),
        F32x8(
            0.0,
            1.0,
            32768.0,
            14348907.0,
            1073741824.0,
            30517578125.0,
            470184984576.0,
            4747561509943.0,
        ),
    )
    assert_almost_equal(
        f32x4_val.__pow__(-1.0), F32x4(inf, 1.0, 0.5, 0.333333333)
    )
    assert_equal(f32x4_val.__pow__(0.0), F32x4(1.0, 1.0, 1.0, 1.0))
    assert_equal(F32x4(1, 1, 1, 1).__pow__(100.0), F32x4(1.0, 1.0, 1.0, 1.0))
    assert_equal(
        String(F32x4(inf, -inf, nan, 1).__pow__(3.0)), "[inf, -inf, nan, 1.0]"
    )
    assert_almost_equal(
        f32x4_val.__pow__(0.5), F32x4(0.0, 1.0, 1.414213562, 1.732050808)
    )

    assert_almost_equal(
        F32x4(1, 2, 3, 4).__pow__(F32x4(2, 3, 2, 1)), F32x4(1.0, 8.0, 9.0, 4.0)
    )

    var f32x4_neg_zero = F32x4(neg_zero, neg_zero, neg_zero, neg_zero)
    assert_equal(
        f32x4_neg_zero.__pow__(F32x4(2.0, 3.0, 1.0, 4.0)),
        F32x4(0.0, 0.0, 0.0, 0.0),
    )
    assert_equal(
        f32x4_neg_zero.__pow__(3.0),
        F32x4(neg_zero, neg_zero, neg_zero, neg_zero),
    )

    assert_almost_equal(
        F32x4(neg_zero, 1.0, 2.0, 3.0).__pow__(F32x4(2.0, 4.0, 8.0, 16.0)),
        F32x4(0.0, 1.0, 256.0, 43046721.0),
    )

    assert_equal(
        F32x4(2.0, 3.0, 4.0, 5.0).__pow__(neg_zero), F32x4(1.0, 1.0, 1.0, 1.0)
    )

    assert_equal(
        String(
            F32x4(inf, neg_inf, nan, 1.0).__pow__(F32x4(2.0, 3.0, 2.0, 0.0))
        ),
        "[inf, -inf, nan, 1.0]",
    )

    assert_equal(
        F32x4(neg_inf, neg_inf, neg_inf, neg_inf).__pow__(
            F32x4(2.0, 3.0, 4.0, 5.0)
        ),
        F32x4(inf, neg_inf, inf, neg_inf),
    )

    # Float64 tests
    alias F64x4 = SIMD[DType.float64, 4]

    assert_equal(
        F64x4(0, 1, 2, 3).__pow__(20.0),
        F64x4(0.0, 1.0, 1048576.0, 3486784401.0),
    )

    assert_almost_equal(
        F64x4(1.0, 2.0, 3.0, 4.0).__pow__(F64x4(2.0, 3.0, 2.0, 1.0)),
        F64x4(1.0, 8.0, 9.0, 4.0),
    )

    # Int32 tests
    alias I32x4 = SIMD[DType.int32, 4]

    var i32x4_val = I32x4(0, 1, 2, 3)

    assert_equal(i32x4_val.__pow__(20), I32x4(0, 1, 1048576, 3486784401))
    assert_equal(i32x4_val.__pow__(0), I32x4(1, 1, 1, 1))
    assert_equal(I32x4(-2, -1, 0, 1).__pow__(3), I32x4(-8, -1, 0, 1))
    assert_equal(
        I32x4(2, 2, 2, 2).__pow__(30),
        I32x4(1073741824, 1073741824, 1073741824, 1073741824),
    )

    assert_equal(
        I32x4(2, 3, 4, 5).__pow__(I32x4(3, 2, 1, 0)), I32x4(8, 9, 4, 1)
    )

    var i32x4_edge_base = I32x4(-2147483648, -1, 0, 2147483647)
    var i32x4_edge_exp = I32x4(31, 31, 31, 31)
    assert_equal(
        i32x4_edge_base.__pow__(i32x4_edge_exp), I32x4(0, -1, 0, 2147483647)
    )
    assert_equal(i32x4_edge_base.__pow__(32), I32x4(0, 1, 0, 1))

    # UInt32 tests
    alias U32x4 = SIMD[DType.uint32, 4]

    var u32x4_val = U32x4(0, 1, 2, 3)

    assert_equal(u32x4_val.__pow__(20), U32x4(0, 1, 1048576, 3486784401))

    assert_equal(
        U32x4(1, 2, 3, 4).__pow__(U32x4(0, 1, 2, 3)), U32x4(1, 2, 9, 64)
    )

    var u32x4_edge_base = U32x4(0, 1, 2147483647, 4294967295)
    assert_equal(
        u32x4_edge_base.__pow__(U32x4(31, 31, 31, 31)),
        U32x4(0, 1, 2147483647, 4294967295),
    )
    assert_equal(u32x4_edge_base.__pow__(32), U32x4(0, 1, 1, 1))

    # Int8 tests
    alias I8x4 = SIMD[DType.int8, 4]

    var i8x4_val = I8x4(0, 1, 2, 3)

    assert_equal(i8x4_val.__pow__(2), I8x4(0, 1, 4, 9))
    assert_equal(i8x4_val.__pow__(7), I8x4(0, 1, 128, -117))
    assert_equal(I8x4(-128, -1, 0, 127).__pow__(3), I8x4(0, -1, 0, 127))

    # UInt8 tests
    alias U8x4 = SIMD[DType.uint8, 4]

    var u8x4_val = U8x4(0, 1, 2, 3)
    assert_equal(u8x4_val.__pow__(2), U8x4(0, 1, 4, 9))
    assert_equal(u8x4_val.__pow__(8), U8x4(0, 1, 0, 161))
    assert_equal(u8x4_val.__pow__(U8x4(3, 5, 7, 9)), U8x4(0, 1, 128, 227))


def test_powf():
    assert_almost_equal(Float32(2.0) ** Float32(0.5), 1.4142135)
    assert_almost_equal(Float32(2.0) ** Float32(-0.5), 0.707107)
    assert_almost_equal(Float32(50.0) ** Float32(2.5), 17677.6695297)
    assert_almost_equal(Float32(12.0) ** Float32(0.4), 2.70192)
    assert_almost_equal(Float32(-1.0) ** Float32(-1), -1)
    assert_almost_equal(Float32(0.001) ** Float32(0.001), 0.99311605)

    assert_almost_equal(Float64(0.001) ** Float64(0.001), 0.99311605)

    assert_almost_equal(Float32(-4) ** Float32(-3), -0.015625)

    assert_almost_equal(
        SIMD[DType.float64, 8](1.0, -1.0, 2.0, -2.0, 4.0, -4.0, -2.0, 3.0)
        ** SIMD[DType.float64, 8](1, 2, 3, 4, 5, 6, 2, 1),
        SIMD[DType.float64, 8](1, 1, 8, 16, 1024, 4096, 4, 3),
    )


def test_rpow():
    alias F32x4 = SIMD[DType.float32, 4]
    alias I32x4 = SIMD[DType.int32, 4]

    var f32x4_val = F32x4(0, 1, 2, 3)
    var i32x4_val = I32x4(0, 1, 2, 3)

    assert_equal(0**i32x4_val, I32x4(1, 0, 0, 0))
    assert_equal(2**i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal((-1) ** i32x4_val, I32x4(1, -1, 1, -1))

    assert_equal(Int(0) ** i32x4_val, I32x4(1, 0, 0, 0))
    assert_equal(Int(2) ** i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal(Int(-1) ** i32x4_val, I32x4(1, -1, 1, -1))

    assert_equal(UInt(2) ** i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal(UInt(0) ** i32x4_val, I32x4(1, 0, 0, 0))

    assert_almost_equal(1.0**f32x4_val, F32x4(1.0, 1.0, 1.0, 1.0))
    assert_almost_equal(2.5**f32x4_val, F32x4(1.0, 2.5, 6.25, 15.625))
    assert_almost_equal(3.0**f32x4_val, F32x4(1.0, 3.0, 9.0, 27.0))


def test_modf():
    var f32 = _modf(Float32(123.5))
    assert_almost_equal(f32[0], 123)
    assert_almost_equal(f32[1], 0.5)

    var f64 = _modf(Float64(123.5))
    assert_almost_equal(f64[0], 123)
    assert_almost_equal(f64[1], 0.5)

    f64 = _modf(Float64(0))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0)

    f64 = _modf(Float64(0.5))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0.5)

    f64 = _modf(Float64(-0.5))
    assert_almost_equal(f64[0], -0)
    assert_almost_equal(f64[1], -0.5)

    f64 = _modf(Float64(-1.5))
    assert_almost_equal(f64[0], -1)
    assert_almost_equal(f64[1], -0.5)


def test_split():
    var tup4 = SIMD[DType.int, 4](1, 2, -3, -4).split()
    assert_equal(tup4[0], __type_of(tup4[0])(1, 2))
    assert_equal(tup4[1], __type_of(tup4[1])(-3, -4))

    var tup8 = SIMD[DType.uint, 8](1, 2, 3, 4, 5, 6, 7, 8).split()
    assert_equal(tup8[0], __type_of(tup8[0])(1, 2, 3, 4))
    assert_equal(tup8[1], __type_of(tup8[1])(5, 6, 7, 8))


def test_contains():
    var x = SIMD[DType.int8, 4](1, 2, 3, 4)
    assert_true(1 in x and 2 in x and 3 in x and 4 in x)
    assert_false(0 in x or 5 in x)
    var y = SIMD[DType.float16, 4](1, 2, 3, 4)
    assert_true(1 in y and 2 in y and 3 in y and 4 in y)
    assert_false(0 in y or 5 in y)


def test_comparison():
    alias dtypes = (
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float16,
        DType.float32,
        DType.float64,
        DType.int,
        DType.uint,
    )

    @parameter
    fn test_dtype[dtype: DType]() raises:
        alias X4 = SIMD[dtype, 4]

        @parameter
        if dtype.is_signed():
            var simd_val = X4(-10, -8, -6, -4)

            assert_true(simd_val == simd_val)
            assert_false(simd_val == X4(0))
            assert_true(simd_val != X4(0))
            assert_false(simd_val != simd_val)
            assert_true(simd_val[0] < simd_val[1])
            assert_true(simd_val[0] <= simd_val[1])
            assert_false(simd_val[2] >= simd_val[3])
            assert_false(simd_val[2] > simd_val[3])

            assert_true(simd_val.lt(X4(-1)).reduce_and())
            assert_false(simd_val.lt(X4(-12)).reduce_or())
            var mixed_lt = simd_val.lt(X4(-6))
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(simd_val.le(X4(-4)).reduce_and())
            assert_false(simd_val.le(X4(-11)).reduce_or())
            var mixed_le = simd_val.le(X4(-8))
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(simd_val.eq(X4(-10, -8, -6, -4)).reduce_and())
            assert_false(simd_val.eq(X4(0)).reduce_or())
            var mixed_eq = simd_val.eq(X4(-10))
            assert_true(mixed_eq[0])
            assert_false(mixed_eq[1])
            assert_false(mixed_eq[2])
            assert_false(mixed_eq[3])

            assert_true(simd_val.ne(X4(0)).reduce_and())
            assert_false(simd_val.ne(X4(-10, -8, -6, -4)).reduce_or())
            var mixed_ne = simd_val.ne(X4(-8))
            assert_true(mixed_ne[0])
            assert_false(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_true(mixed_ne[3])

            assert_true(simd_val.gt(X4(-11)).reduce_and())
            assert_false(simd_val.gt(X4(-1)).reduce_or())
            var mixed_gt = simd_val.gt(X4(-6))
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_false(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(simd_val.ge(X4(-10)).reduce_and())
            assert_false(simd_val.ge(X4(-1)).reduce_or())
            var mixed_ge = simd_val.ge(X4(-6))
            assert_false(mixed_ge[0])
            assert_false(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

        @parameter
        if dtype.is_numeric():
            var simd_val = X4(1, 2, 3, 4)

            assert_true(simd_val == simd_val)
            assert_false(simd_val == X4(0))
            assert_true(simd_val != X4(0))
            assert_false(simd_val != simd_val)
            assert_true(simd_val[0] < simd_val[1])
            assert_true(simd_val[0] <= simd_val[1])
            assert_false(simd_val[2] >= simd_val[3])
            assert_false(simd_val[2] > simd_val[3])

            assert_true(simd_val.lt(X4(5)).reduce_and())
            assert_false(simd_val.lt(X4(0)).reduce_or())
            var mixed_lt = simd_val.lt(X4(3))
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(simd_val.le(X4(4)).reduce_and())
            assert_false(simd_val.le(X4(0)).reduce_or())
            var mixed_le = simd_val.le(X4(3))
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_true(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(simd_val.eq(X4(1, 2, 3, 4)).reduce_and())
            assert_false(simd_val.eq(X4(5)).reduce_or())
            var mixed_eq = simd_val.eq(X4(1))
            assert_true(mixed_eq[0])
            assert_false(mixed_eq[1])
            assert_false(mixed_eq[2])
            assert_false(mixed_eq[3])

            assert_true(simd_val.ne(X4(5)).reduce_and())
            assert_false(simd_val.ne(X4(1, 2, 3, 4)).reduce_or())
            var mixed_ne = simd_val.ne(X4(4))
            assert_true(mixed_ne[0])
            assert_true(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_false(mixed_ne[3])

            assert_true(simd_val.gt(X4(0)).reduce_and())
            assert_false(simd_val.gt(X4(4)).reduce_or())
            var mixed_gt = simd_val.gt(X4(2))
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_true(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(simd_val.ge(X4(1)).reduce_and())
            assert_false(simd_val.ge(X4(5)).reduce_or())
            var mixed_ge = simd_val.ge(X4(2))
            assert_false(mixed_ge[0])
            assert_true(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

        @parameter
        if dtype is DType.bool:
            var all_true = SIMD[DType.bool, 4](fill=True)
            var all_false = SIMD[DType.bool, 4](fill=False)
            var mixed = SIMD[DType.bool, 4](True, True, False, False)

            assert_true(all_true == all_true)
            assert_false(all_true == all_false)
            assert_true(all_true != all_false)
            assert_false(all_true != all_true)
            assert_false(mixed[0] < mixed[1])
            assert_true(mixed[0] <= mixed[1])
            assert_true(mixed[2] >= mixed[3])
            assert_false(mixed[2] > mixed[3])

            assert_true(all_false.lt(all_true).reduce_and())
            assert_false(all_true.lt(all_false).reduce_or())
            var mixed_lt = all_false.lt(mixed)
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(all_false.le(all_true).reduce_and())
            assert_false(all_true.le(all_false).reduce_or())
            var mixed_le = all_true.le(mixed)
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(
                all_true.eq(
                    SIMD[DType.bool, 4](True, True, True, True)
                ).reduce_and()
            )
            assert_false(all_true.eq(all_false).reduce_or())
            var mixed_eq = all_true.eq(mixed)
            assert_true(mixed_eq[0])
            assert_true(mixed_eq[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(all_true.ne(all_false).reduce_and())
            assert_false(
                all_true.ne(
                    SIMD[DType.bool, 4](True, True, True, True)
                ).reduce_or()
            )
            var mixed_ne = all_true.ne(mixed)
            assert_false(mixed_ne[0])
            assert_false(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_true(mixed_ne[3])

            assert_true(all_true.gt(all_false).reduce_and())
            assert_false(all_false.gt(all_true).reduce_or())
            var mixed_gt = all_true.gt(mixed)
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_true(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(all_true.ge(all_false).reduce_and())
            assert_false(all_false.ge(all_true).reduce_or())
            var mixed_ge = all_true.ge(mixed)
            assert_true(mixed_ge[0])
            assert_true(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

    @parameter
    for i in range(dtypes.__len__()):
        alias dtype = dtypes[i]
        test_dtype[dtype]()

    test_dtype[DType.bfloat16]()


def test_float_conversion():
    assert_almost_equal(Float64(Int32(45)), 45.0)
    assert_almost_equal(Float64(Float32(34.32)), 34.32)
    assert_almost_equal(Float64(UInt64(36)), 36.0)


def test_from_bytes_as_bytes():
    # Test scalar types with specific byte patterns
    alias TwoBytes = InlineArray[Byte, size_of[Int16]()]
    alias TwoUBytes = InlineArray[Byte, size_of[UInt16]()]
    alias FourBytes = InlineArray[Byte, size_of[Int32]()]

    assert_equal(Int16.from_bytes[big_endian=True](TwoBytes(0, 16)), 16)
    assert_equal(Int16.from_bytes[big_endian=False](TwoBytes(0, 16)), 4096)
    assert_equal(Int16.from_bytes[big_endian=True](TwoBytes(252, 0)), -1024)
    assert_equal(UInt16.from_bytes[big_endian=True](TwoUBytes(252, 0)), 64512)
    assert_equal(Int16.from_bytes[big_endian=False](TwoBytes(252, 0)), 252)
    assert_equal(Int32.from_bytes[big_endian=True](FourBytes(0, 0, 0, 1)), 1)
    assert_equal(
        Int32.from_bytes[big_endian=False](FourBytes(0, 0, 0, 1)),
        16777216,
    )
    assert_equal(
        Int32.from_bytes[big_endian=True](FourBytes(1, 0, 0, 0)),
        16777216,
    )
    assert_equal(
        Int32.from_bytes[big_endian=True](FourBytes(1, 0, 0, 1)),
        16777217,
    )
    assert_equal(
        Int32.from_bytes[big_endian=False](FourBytes(1, 0, 0, 1)),
        16777217,
    )
    assert_equal(
        Int32.from_bytes[big_endian=True](FourBytes(255, 0, 0, 0)),
        -16777216,
    )

    # Test scalar roundtrip conversions
    for x in List[Int16](10, 100, -12, 0, 1, -1, 1000, -1000):

        @parameter
        for b in range(2):
            assert_equal(
                Int16.from_bytes[big_endian=b](
                    Int16(x).as_bytes[big_endian=b]()
                ),
                x,
            )

    # Test SIMD vector roundtrip conversions (from test_comprehensive_from_bytes)
    var original_int32 = SIMD[DType.int32, 2](1, 2)

    # Test little endian roundtrip
    var bytes_le = original_int32.as_bytes[big_endian=False]()
    var int32_from_le = SIMD[DType.int32, 2].from_bytes[big_endian=False](
        bytes_le
    )
    assert_equal(int32_from_le, original_int32)

    # Test big endian roundtrip
    var bytes_be = original_int32.as_bytes[big_endian=True]()
    var int32_from_be = SIMD[DType.int32, 2].from_bytes[big_endian=True](
        bytes_be
    )
    assert_equal(int32_from_be, original_int32)

    # Test with float64 roundtrip (using default endianness)
    var original_float64 = SIMD[DType.float64, 2](1.0, 2.0)
    var float_bytes = original_float64.as_bytes()
    var float64_from_bytes = SIMD[DType.float64, 2].from_bytes(float_bytes)
    assert_equal(float64_from_bytes, original_float64)

    # Test as_bytes conversion with specific byte patterns (from test_comprehensive_as_bytes)
    var int32_vals = SIMD[DType.int32, 2](0x12345678, 0x87654321)
    var bytes_le_vals = int32_vals.as_bytes[big_endian=False]()
    var bytes_be_vals = int32_vals.as_bytes[big_endian=True]()

    # Little endian: least significant byte first
    assert_equal(bytes_le_vals[0], 0x78)
    assert_equal(bytes_le_vals[1], 0x56)
    assert_equal(bytes_le_vals[2], 0x34)
    assert_equal(bytes_le_vals[3], 0x12)

    # Big endian: most significant byte first
    assert_equal(bytes_be_vals[0], 0x12)
    assert_equal(bytes_be_vals[1], 0x34)
    assert_equal(bytes_be_vals[2], 0x56)
    assert_equal(bytes_be_vals[3], 0x78)

    # Test roundtrip conversion
    var reconstructed_le = SIMD[DType.int32, 2].from_bytes[big_endian=False](
        bytes_le_vals
    )
    var reconstructed_be = SIMD[DType.int32, 2].from_bytes[big_endian=True](
        bytes_be_vals
    )
    assert_equal(reconstructed_le, int32_vals)
    assert_equal(reconstructed_be, int32_vals)


def test_vector_from_bytes_as_bytes():
    # Test various SIMD vector types with comprehensive byte conversions
    var v8_u8 = SIMD[DType.uint8, 8](1, 2, 3, 4, 5, 6, 7, 8)
    assert_equal(v8_u8, SIMD[DType.uint8, 8].from_bytes(v8_u8.as_bytes()))

    var v8_u16 = SIMD[DType.uint16, 8](1, 2, 3, 4, 5, 6, 7, 8)
    # fmt: off
    var expected_v8_u16_be_bytes = [
        0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
    ]
    # fmt: on
    var actual_v8_u16_be_bytes = v8_u16.as_bytes[big_endian=True]()
    for i, expected in enumerate(expected_v8_u16_be_bytes):
        assert_equal(Int(actual_v8_u16_be_bytes[i]), expected)
    # fmt: off
    var expected_v8_u16_le_bytes = [
        1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
    ]
    # fmt: on
    var actual_v8_u16_le_bytes = v8_u16.as_bytes[big_endian=False]()
    for i, expected in enumerate(expected_v8_u16_le_bytes):
        assert_equal(Int(actual_v8_u16_le_bytes[i]), expected)

    var v8_i64 = SIMD[DType.int64, 8](1, -2, 3, -4, 5, -6, 7, -8)
    assert_equal(
        v8_i64,
        SIMD[DType.int64, 8].from_bytes[big_endian=False](
            v8_i64.as_bytes[big_endian=False]()
        ),
    )
    assert_equal(
        v8_i64,
        SIMD[DType.int64, 8].from_bytes[big_endian=True](
            v8_i64.as_bytes[big_endian=True]()
        ),
    )

    var v8_f64 = SIMD[DType.float64, 8](
        1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8
    )
    assert_equal(v8_f64, SIMD[DType.float64, 8].from_bytes(v8_f64.as_bytes()))

    var v8_bool = SIMD[DType.bool, 8](
        True, True, False, True, False, True, True, True
    )
    assert_equal(v8_bool, SIMD[DType.bool, 8].from_bytes(v8_bool.as_bytes()))


def test_reversed():
    fn test[dtype: DType]() raises:
        assert_equal(
            SIMD[dtype, 4](1, 2, 3, 4).reversed(), SIMD[dtype, 4](4, 3, 2, 1)
        )

    test[DType.uint8]()
    test[DType.uint16]()
    test[DType.uint32]()
    test[DType.uint64]()
    test[DType.int8]()
    test[DType.int16]()
    test[DType.int32]()
    test[DType.int64]()
    test[DType.float16]()
    test[DType.float32]()
    test[DType.float64]()


def test_large_int_types():
    var x = Int128(1234567890)
    var y = UInt128(1234567890)
    var z = Int256(1234567890)
    var w = UInt256(1234567890)

    assert_equal(x, 1234567890)
    assert_equal(y, 1234567890)
    assert_equal(z, 1234567890)
    assert_equal(w, 1234567890)

    assert_equal(x.cast[DType.uint128](), y)
    assert_equal(x.cast[DType.int256](), z)
    assert_equal(x.cast[DType.uint256](), w)

    assert_equal(y.cast[DType.int128](), x)
    assert_equal(y.cast[DType.int256](), z)
    assert_equal(y.cast[DType.uint256](), w)

    assert_equal(z.cast[DType.int128](), x)
    assert_equal(z.cast[DType.uint128](), y)
    assert_equal(z.cast[DType.uint256](), w)

    assert_equal(w.cast[DType.int128](), x)
    assert_equal(w.cast[DType.uint128](), y)
    assert_equal(w.cast[DType.int256](), z)

    assert_equal(x.cast[DType.uint128]() + y, y + y)
    assert_equal(x.cast[DType.int256]() + z, z + z)


def test_is_power_of_two():
    # Test comprehensive cases with known powers of two
    var powers = SIMD[DType.uint32, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var power_results = powers.is_power_of_two()
    var expected_powers = SIMD[DType.bool, 8](
        True, True, True, True, True, True, True, True
    )
    assert_equal(power_results, expected_powers)

    # Test non-powers of two including zero and common edge cases
    var non_powers = SIMD[DType.uint32, 8](0, 3, 5, 6, 7, 9, 10, 15)
    var non_power_results = non_powers.is_power_of_two()
    var expected_non_powers = SIMD[DType.bool, 8](
        False, False, False, False, False, False, False, False
    )
    assert_equal(non_power_results, expected_non_powers)

    # Test with different integer types and larger powers (avoiding duplicate zero tests)
    # Note that for DType.int8, the maximum value is 127, so 2**7 == 128 which overflows.
    alias var1 = SIMD[DType.int8, 4](-114, 100, 2**6, 2**7)
    assert_equal(
        var1.is_power_of_two(),
        SIMD[DType.bool, 4](False, False, True, False),
    )

    alias var2 = SIMD[DType.int16, 4](-11444, 3000, 2**13, 2**14)
    assert_equal(
        var2.is_power_of_two(),
        SIMD[DType.bool, 4](False, False, True, True),
    )

    alias var3 = SIMD[DType.int32, 4](-111444, 30000, 2**29, 2**30)
    assert_equal(
        var3.is_power_of_two(),
        SIMD[DType.bool, 4](False, False, True, True),
    )

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[DType.int64, 4](-111444444, 3000000, 2**59, 2**60)
    alias var4 = SIMD[DType.int64, 4](
        -111444444, 3000000, 576460752303423488, 1152921504606846976
    )
    assert_equal(
        var4.is_power_of_two(),
        SIMD[DType.bool, 4](False, False, True, True),
    )

    # Test edge cases: negative numbers and boundary values
    var signed_edge_cases = SIMD[DType.int32, 4](-4, -1, Int32.MAX, 2**31)
    var signed_results = signed_edge_cases.is_power_of_two()
    var expected_signed = SIMD[DType.bool, 4](False, False, False, False)
    assert_equal(signed_results, expected_signed)

    assert_equal(Int64.MIN.is_power_of_two(), False)


def test_comptime():
    alias v = Int32(0b1111_1111)
    alias n = count_leading_zeros(v)
    # Verify that count_leading_zeros works at comptime.
    assert_equal(n, 24)


def test_fma():
    # Test fused multiply-add operation: self * multiplier + accumulator
    var simd = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var multiplier = SIMD[DType.float32, 4](2.0, 3.0, 4.0, 5.0)
    var accumulator = SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0)

    var result = simd.fma(multiplier, accumulator)
    var expected = SIMD[DType.float32, 4](
        3.0, 7.0, 13.0, 21.0
    )  # (1*2+1, 2*3+1, 3*4+1, 4*5+1)

    assert_equal(result, expected)

    # Test with integer types
    var int_simd = SIMD[DType.int32, 4](1, 2, 3, 4)
    var int_mult = SIMD[DType.int32, 4](2, 3, 4, 5)
    var int_acc = SIMD[DType.int32, 4](1, 1, 1, 1)

    var int_result = int_simd.fma(int_mult, int_acc)
    var int_expected = SIMD[DType.int32, 4](3, 7, 13, 21)

    assert_equal(int_result, int_expected)


def test_slice():
    # Test slicing SIMD vectors
    var simd8 = SIMD[DType.int32, 8](1, 2, 3, 4, 5, 6, 7, 8)

    # Test slice with default offset (0)
    var slice4_0 = simd8.slice[4]()
    var expected4_0 = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(slice4_0, expected4_0)

    # Test slice with offset
    var slice4_2 = simd8.slice[4, offset=2]()
    var expected4_2 = SIMD[DType.int32, 4](3, 4, 5, 6)
    assert_equal(slice4_2, expected4_2)

    # Test slice with width 2
    var slice2_3 = simd8.slice[2, offset=3]()
    var expected2_3 = SIMD[DType.int32, 2](4, 5)
    assert_equal(slice2_3, expected2_3)

    # Test with scalar (width 1)
    var slice1_5 = simd8.slice[1, offset=5]()
    var expected1_5 = Int32(6)
    assert_equal(slice1_5, expected1_5)


def test_hash():
    # Test hash function for SIMD values
    var simd1 = SIMD[DType.int32, 4](1, 2, 3, 4)
    var simd2 = SIMD[DType.int32, 4](1, 2, 3, 4)  # Same values
    var simd3 = SIMD[DType.int32, 4](1, 2, 3, 5)  # Different last value

    # Same values should have same hash
    assert_equal(hash(simd1), hash(simd2))

    # Different values should typically have different hashes (not guaranteed but likely)
    assert_true(hash(simd1) != hash(simd3))

    # Test with different types
    var float_simd = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var float_hash = hash(float_simd)
    # Hash should be consistent for same values
    assert_equal(hash(float_simd), float_hash)


def test_reduce_bitwise_ops():
    # Test reduce_and
    var all_ones = SIMD[DType.uint8, 4](0xFF, 0xFF, 0xFF, 0xFF)
    var mixed_bits = SIMD[DType.uint8, 4](0xFF, 0xF0, 0x0F, 0xFF)

    assert_equal(all_ones.reduce_and(), UInt8(0xFF))
    assert_equal(
        mixed_bits.reduce_and(), UInt8(0x00)
    )  # 0xFF & 0xF0 & 0x0F & 0xFF = 0x00

    # Test reduce_or
    var all_zeros = SIMD[DType.uint8, 4](0x00, 0x00, 0x00, 0x00)
    var some_bits = SIMD[DType.uint8, 4](0x01, 0x02, 0x04, 0x08)

    assert_equal(all_zeros.reduce_or(), UInt8(0x00))
    assert_equal(
        some_bits.reduce_or(), UInt8(0x0F)
    )  # 0x01 | 0x02 | 0x04 | 0x08 = 0x0F

    # Test with larger vectors
    var large_and = SIMD[DType.uint16, 8](
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF
    )
    assert_equal(large_and.reduce_and(), UInt16(0xFFFF))

    var pattern_or = SIMD[DType.uint16, 8](
        0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080
    )
    assert_equal(pattern_or.reduce_or(), UInt16(0x00FF))


def test_float_literal_init():
    # Test initialization from FloatLiteral
    var float_simd = SIMD[DType.float32, 4](3.14159)
    assert_almost_equal(float_simd[0], Float32(3.14159), atol=1e-5)
    assert_almost_equal(float_simd[1], Float32(3.14159), atol=1e-5)
    assert_almost_equal(float_simd[2], Float32(3.14159), atol=1e-5)
    assert_almost_equal(float_simd[3], Float32(3.14159), atol=1e-5)

    # Test with double precision
    var double_simd = SIMD[DType.float64, 2](2.718281828459045)
    assert_almost_equal(double_simd[0], Float64(2.718281828459045), atol=1e-15)
    assert_almost_equal(double_simd[1], Float64(2.718281828459045), atol=1e-15)

    # Test with various float types
    var f16_simd = SIMD[DType.float16, 4](1.5)
    # Note: float16 has limited precision
    assert_almost_equal(
        Float32(Float32(f16_simd[0])),
        Float32(1.5),
        atol=1e-3,
    )


def test_int_literal_init():
    assert_equal(UInt8(255), UInt8(-1))
    assert_equal(UInt8(256), UInt8(0))
    assert_equal(UInt16(65535), UInt16(-1))
    assert_equal(UInt16(65536), UInt16(0))
    assert_equal(UInt32(4294967295), UInt32(-1))
    assert_equal(UInt32(4294967296), UInt32(0))
    assert_equal(UInt64(18446744073709551615), UInt64(-1))
    assert_equal(UInt64(18446744073709551616), UInt64(0))

    assert_equal(Int8(-128), Int8(128))
    assert_equal(Int8(-129), Int8(127))
    assert_equal(Int16(-32768), Int16(32768))
    assert_equal(Int16(-32769), Int16(32767))
    assert_equal(Int32(-2147483648), Int32(2147483648))
    assert_equal(Int32(-2147483649), Int32(2147483647))
    assert_equal(Int64(-9223372036854775808), Int64(9223372036854775808))
    assert_equal(Int64(-9223372036854775809), Int64(9223372036854775807))

    alias Index = Scalar[DType.int]
    alias UIndex = Scalar[DType.uint]

    @parameter
    if is_64bit():
        assert_equal(Index(-9223372036854775808), Index(9223372036854775808))
        assert_equal(Index(-9223372036854775809), Index(9223372036854775807))
        assert_equal(UIndex(0), UIndex(18446744073709551616))
        assert_equal(UIndex(-1), UIndex(18446744073709551615))

        alias i1 = Index(-9223372036854775808)
        alias i2 = Index(9223372036854775808)
        assert_equal(i1, i2)
        alias ui1 = UIndex(-1)
        alias ui2 = UIndex(18446744073709551615)
        assert_equal(ui1, ui2)
    else:
        assert_equal(Index(-2147483648), Index(2147483648))
        assert_equal(Index(-2147483649), Index(2147483647))
        assert_equal(UIndex(0), UIndex(4294967296))
        assert_equal(UIndex(-1), UIndex(4294967295))

        alias i1 = Index(-2147483648)
        alias i2 = Index(2147483648)
        assert_equal(i1, i2)
        alias ui1 = UIndex(-1)
        alias ui2 = UIndex(4294967295)
        assert_equal(ui1, ui2)


def test_bool_init():
    # Test initialization from Bool for boolean SIMD types
    var bool_simd = SIMD[DType.bool, 4](fill=True)
    assert_equal(bool_simd, SIMD[DType.bool, 4](True, True, True, True))

    var bool_simd_false = SIMD[DType.bool, 4](fill=False)
    assert_equal(
        bool_simd_false, SIMD[DType.bool, 4](False, False, False, False)
    )

    # Test mixed boolean initialization
    var mixed_bool = SIMD[DType.bool, 4](True, False, True, False)
    assert_true(mixed_bool[0])
    assert_false(mixed_bool[1])
    assert_true(mixed_bool[2])
    assert_false(mixed_bool[3])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
    # TODO: add tests for __and__, __or__, and comparison operators
