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

from sys.info import CompilationTarget, is_64bit

from testing import assert_almost_equal, assert_equal, assert_false, assert_true

from utils.numerics import (
    FPUtils,
    get_accum_type,
    isfinite,
    isinf,
    isnan,
    max_or_inf,
    min_or_neg_inf,
    nextafter,
)


# TODO: improve coverage and organization of these tests
def test_FPUtils():
    assert_equal(FPUtils[DType.float32].mantissa_width(), 23)
    assert_equal(FPUtils[DType.float32].exponent_bias(), 127)

    alias FPU64 = FPUtils[DType.float64]

    assert_equal(FPU64.mantissa_width(), 52)
    assert_equal(FPU64.exponent_bias(), 1023)

    assert_equal(FPU64.get_exponent(FPU64.set_exponent(1, 2)), 2)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(1, 3)), 3)
    assert_equal(FPU64.get_exponent(FPU64.set_exponent(-1, 4)), 4)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(-1, 5)), 5)
    assert_true(FPU64.get_sign(FPU64.set_sign(0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(0, False)))
    assert_true(FPU64.get_sign(FPU64.set_sign(-0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(-0, False)))
    assert_false(FPU64.get_sign(1))
    assert_true(FPU64.get_sign(-1))
    assert_false(FPU64.get_sign(FPU64.pack(False, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(False, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(False, 6, 12)), 12)
    assert_true(FPU64.get_sign(FPU64.pack(True, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(True, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(True, 6, 12)), 12)


def test_get_accum_type():
    assert_equal(get_accum_type[DType.float32](), DType.float32)
    assert_equal(get_accum_type[DType.float64](), DType.float64)
    assert_equal(get_accum_type[DType.bfloat16](), DType.float32)
    assert_equal(get_accum_type[DType.int8](), DType.int8)
    assert_equal(get_accum_type[DType.int16](), DType.int16)
    assert_equal(get_accum_type[DType.int32](), DType.int32)
    assert_equal(get_accum_type[DType.int64](), DType.int64)
    assert_equal(get_accum_type[DType.uint8](), DType.uint8)
    assert_equal(get_accum_type[DType.uint16](), DType.uint16)
    assert_equal(get_accum_type[DType.uint32](), DType.uint32)
    assert_equal(get_accum_type[DType.uint64](), DType.uint64)


def test_isfinite():
    assert_true(isfinite(Float32(33)))

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not CompilationTarget.has_neon():
        assert_false(isfinite(DType.inf[DType.bfloat16]()))
        assert_false(isfinite(DType.neg_inf[DType.bfloat16]()))
        assert_false(isfinite(DType.nan[DType.bfloat16]()))

    assert_false(isfinite(DType.inf[DType.float16]()))
    assert_false(isfinite(DType.inf[DType.float32]()))
    assert_false(isfinite(DType.inf[DType.float64]()))
    assert_false(isfinite(DType.neg_inf[DType.float16]()))
    assert_false(isfinite(DType.neg_inf[DType.float32]()))
    assert_false(isfinite(DType.neg_inf[DType.float64]()))
    assert_false(isfinite(DType.nan[DType.float16]()))
    assert_false(isfinite(DType.nan[DType.float32]()))
    assert_false(isfinite(DType.nan[DType.float64]()))


def test_isinf():
    assert_false(isinf(Float32(33)))

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not CompilationTarget.has_neon():
        assert_true(isinf(DType.inf[DType.bfloat16]()))
        assert_true(isinf(DType.neg_inf[DType.bfloat16]()))
        assert_false(isinf(DType.nan[DType.bfloat16]()))

    assert_true(isinf(DType.inf[DType.float16]()))
    assert_true(isinf(DType.inf[DType.float32]()))
    assert_true(isinf(DType.inf[DType.float64]()))
    assert_true(isinf(DType.neg_inf[DType.float16]()))
    assert_true(isinf(DType.neg_inf[DType.float32]()))
    assert_true(isinf(DType.neg_inf[DType.float64]()))
    assert_false(isinf(DType.nan[DType.float16]()))
    assert_false(isinf(DType.nan[DType.float32]()))
    assert_false(isinf(DType.nan[DType.float64]()))


def test_isnan():
    assert_false(isnan(Float32(33)))

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not CompilationTarget.has_neon():
        assert_false(isnan(DType.inf[DType.bfloat16]()))
        assert_false(isnan(DType.neg_inf[DType.bfloat16]()))
        assert_true(isnan(DType.nan[DType.bfloat16]()))

    assert_false(isnan(DType.inf[DType.float16]()))
    assert_false(isnan(DType.inf[DType.float32]()))
    assert_false(isnan(DType.inf[DType.float64]()))
    assert_false(isnan(DType.neg_inf[DType.float16]()))
    assert_false(isnan(DType.neg_inf[DType.float32]()))
    assert_false(isnan(DType.neg_inf[DType.float64]()))
    assert_true(isnan(DType.nan[DType.float16]()))
    assert_true(isnan(DType.nan[DType.float32]()))
    assert_true(isnan(DType.nan[DType.float64]()))


fn overflow_int[dtype: DType]() -> Bool:
    constrained[
        dtype.is_integral(), "comparison only valid on integral types"
    ]()
    return DType.max_finite[dtype]() + 1 < DType.max_finite[dtype]()


fn overflow_fp[dtype: DType]() -> Bool:
    constrained[
        dtype.is_floating_point(),
        "comparison only valid on floating point types",
    ]()
    return DType.max_finite[dtype]() + 1 == DType.max_finite[dtype]()


def test_max_finite():
    assert_almost_equal(DType.max_finite[DType.float32](), 3.4028235e38)
    assert_almost_equal(
        DType.max_finite[DType.float64](), 1.7976931348623157e308
    )

    assert_true(DType.max_finite[DType.bool]())

    assert_true(overflow_int[DType.int8]())
    assert_true(overflow_int[DType.uint8]())
    assert_true(overflow_int[DType.int16]())
    assert_true(overflow_int[DType.uint16]())
    assert_true(overflow_int[DType.int32]())
    assert_true(overflow_int[DType.uint32]())
    assert_true(overflow_int[DType.int64]())
    assert_true(overflow_int[DType.uint64]())
    assert_true(overflow_int[DType.index]())

    assert_true(overflow_fp[DType.float32]())
    assert_true(overflow_fp[DType.float64]())

    assert_equal(DType.max_finite[DType.int8](), 127)
    assert_equal(DType.max_finite[DType.uint8](), 255)
    assert_equal(DType.max_finite[DType.int16](), 32767)
    assert_equal(DType.max_finite[DType.uint16](), 65535)
    assert_equal(DType.max_finite[DType.int32](), 2147483647)
    assert_equal(DType.max_finite[DType.uint32](), 4294967295)
    assert_equal(DType.max_finite[DType.int64](), 9223372036854775807)
    assert_equal(DType.max_finite[DType.uint64](), 18446744073709551615)
    # FIXME(#5214): uncomment once it is closed
    # assert_equal(
    #     DType.max_finite[DType.int128](), 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF
    # )
    # assert_equal(
    #     DType.max_finite[DType.uint128](), 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF
    # )
    # assert_equal(
    #     DType.max_finite[DType.int256](),
    #     0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
    # )
    # assert_equal(
    #     DType.max_finite[DType.uint256](),
    #     0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
    # )

    @parameter
    if is_64bit():
        assert_equal(DType.max_finite[DType.index](), 9223372036854775807)
    else:
        assert_equal(DType.max_finite[DType.index](), 2147483647)


fn underflow_int[dtype: DType]() -> Bool:
    constrained[
        dtype.is_integral(), "comparison only valid on integral types"
    ]()
    return DType.min_finite[dtype]() - 1 > DType.min_finite[dtype]()


fn underflow_fp[dtype: DType]() -> Bool:
    constrained[
        dtype.is_floating_point(),
        "comparison only valid on floating point types",
    ]()
    return DType.min_finite[dtype]() - 1 == DType.min_finite[dtype]()


def test_min_finite():
    assert_almost_equal(DType.min_finite[DType.float32](), -3.4028235e38)
    assert_almost_equal(
        DType.min_finite[DType.float64](), -1.7976931348623157e308
    )

    assert_false(DType.min_finite[DType.bool]())

    assert_true(underflow_int[DType.int8]())
    assert_true(underflow_int[DType.uint8]())
    assert_true(underflow_int[DType.int16]())
    assert_true(underflow_int[DType.uint16]())
    assert_true(underflow_int[DType.int32]())
    assert_true(underflow_int[DType.uint32]())
    assert_true(underflow_int[DType.int64]())
    assert_true(underflow_int[DType.uint64]())
    assert_true(underflow_int[DType.index]())

    assert_true(underflow_fp[DType.float32]())
    assert_true(underflow_fp[DType.float64]())

    assert_equal(DType.min_finite[DType.int8](), -128)
    assert_equal(DType.min_finite[DType.int16](), -32768)
    assert_equal(DType.min_finite[DType.int32](), -2147483648)
    assert_equal(DType.min_finite[DType.int64](), -9223372036854775808)
    # FIXME(#5214): uncomment once it is closed
    # assert_equal(
    #     DType.min_finite[DType.int128](),
    #     -0x8000_0000_0000_0000_0000_0000_0000_0000,
    # )
    # assert_equal(
    #     DType.min_finite[DType.int256](),
    #     -0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000,
    # )

    @parameter
    if is_64bit():
        assert_equal(DType.min_finite[DType.index](), -9223372036854775808)
    else:
        assert_equal(DType.min_finite[DType.index](), -2147483648)


def test_max_or_inf():
    assert_almost_equal(max_or_inf[DType.float32](), DType.inf[DType.float32]())
    assert_almost_equal(max_or_inf[DType.float64](), DType.inf[DType.float64]())
    assert_true(max_or_inf[DType.bool]())


def test_min_or_neg_inf():
    assert_almost_equal(
        min_or_neg_inf[DType.float32](), DType.neg_inf[DType.float32]()
    )
    assert_almost_equal(
        min_or_neg_inf[DType.float64](), DType.neg_inf[DType.float64]()
    )
    assert_false(min_or_neg_inf[DType.bool]())


def test_neg_inf():
    assert_false(isfinite(DType.neg_inf[DType.float32]()))
    assert_false(isfinite(DType.neg_inf[DType.float64]()))
    assert_true(isinf(DType.neg_inf[DType.float32]()))
    assert_true(isinf(DType.neg_inf[DType.float64]()))
    assert_false(isnan(DType.neg_inf[DType.float32]()))
    assert_false(isnan(DType.neg_inf[DType.float64]()))
    assert_equal(-DType.inf[DType.float32](), DType.neg_inf[DType.float32]())
    assert_equal(-DType.inf[DType.float64](), DType.neg_inf[DType.float64]())


def test_nextafter():
    assert_true(
        isnan(nextafter(DType.nan[DType.float32](), DType.nan[DType.float32]()))
    )
    assert_true(
        isinf(nextafter(DType.inf[DType.float32](), DType.inf[DType.float32]()))
    )
    assert_true(
        isinf(
            nextafter(-DType.inf[DType.float32](), -DType.inf[DType.float32]())
        )
    )
    assert_almost_equal(nextafter(Float64(0), Float64(0)), 0)
    assert_almost_equal(nextafter(Float64(0), Float64(1)), 5e-324)
    assert_almost_equal(nextafter(Float64(0), Float64(-1)), -5e-324)
    assert_almost_equal(nextafter(Float64(1), Float64(0)), 0.99999999999999988)
    assert_almost_equal(
        nextafter(Float64(-1), Float64(0)), -0.99999999999999988
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](0, 1)),
        SIMD[DType.float64, 2](0, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](1, 1)),
        SIMD[DType.float64, 2](5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](-1, 1)),
        SIMD[DType.float64, 2](-5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](1, 1), SIMD[DType.float64, 2](0, 0)),
        SIMD[DType.float64, 2](0.99999999999999988, 0.99999999999999988),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](-1, -1), SIMD[DType.float64, 2](0, 0)),
        SIMD[DType.float64, 2](-0.99999999999999988, -0.99999999999999988),
    )


def main():
    test_FPUtils()
    test_get_accum_type()
    test_isfinite()
    test_isinf()
    test_isnan()
    test_max_finite()
    test_max_or_inf()
    test_min_finite()
    test_min_or_neg_inf()
    test_neg_inf()
    test_nextafter()
