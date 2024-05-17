# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn main():
    print(Float64(1.1))  # CHECK: 1.10000{{.*}}
    print(Float64(0.1))  # CHECK-NEXT: 0.10000{{.*}}
    print(Float64(1.0))  # CHECK-NEXT: 1
    print(Float64(1e2))  # CHECK-NEXT: 100
    print(Float64(1.1e2))  # CHECK-NEXT: 110
    print(Float64(0.1e2))  # CHECK-NEXT: 10
    print(Float64(1.0e2))  # CHECK-NEXT: 100
    print(Float64(1e2))  # CHECK-NEXT: 100
    print(Float64(1.1e-2))  # CHECK-NEXT: 0.01099{{.*}}
    print(Float64(0.1e2))  # CHECK-NEXT: 10
    print(Float64(1.0e-2))  # CHECK-NEXT: 0.01
    print(Float64(0.1))  # CHECK-NEXT: 0.100000{{.*}}
    print(Float64(0.0))  # CHECK-NEXT: 0.0
    print(Float64(0e2))  # CHECK-NEXT: 0.0
    print(Float64(0.1e2))  # CHECK-NEXT: 10
    print(Float64(0.0e2))  # CHECK-NEXT: 0.0
    print(Float64(0e2))  # CHECK-NEXT: 0.0
    print(Float64(0.1e-2))  # CHECK-NEXT: 0.001
    print(Float64(0.0e-2))  # CHECK-NEXT: 0.0
    print(Float64(12.31e11))  # CHECK-NEXT: 1231000000000.0
    print(Float64(12.31e-3))  # CHECK-NEXT: 0.01231
    # Check gradual loss of precision for subnormal numbers when
    # converting from infinite precision literal to Float64.
    print(Float64(1.1234567e-305))  # CHECK-NEXT: 1.1234567e-305
    print(Float64(1.1234567e-310))  # CHECK-NEXT: 1.1234567000000234e-310
    print(Float64(1.1234567e-315))  # CHECK-NEXT: 1.1234567021086955e-315
    print(Float64(1.1234567e-320))  # CHECK-NEXT: 1.1235052786429946e-320
    print(Float64(1.1234567e-322))  # CHECK-NEXT: 1.1363509854348671e-322
    print(Float64(1.1234567e-323))  # CHECK-NEXT: 9.8813129168249309e-324
    print(Float64(1.1234567e-324))  # CHECK-NEXT: 0.0
    print(999e9000)  # CHECK-NEXT: inf
    print(-999e9000)  # CHECK-NEXT: -inf

    print(FloatLiteral.infinity)  # CHECK-NEXT: inf
    print(FloatLiteral.negative_infinity)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_zero)  # CHECK-NEXT: -0.0
    print(FloatLiteral.nan)  # CHECK-NEXT: nan

    print(-1 * 0.0)  # CHECK-NEXT: -0.0
    print(-1 * -0.0)  # CHECK-NEXT: 0.0

    # Very special cases of float literals were rounding incorrectly:
    # Specifically, where long division would end due to the mantissa being
    # sufficiently long (including rounding bits), with the least significant
    # (non-rounding) bit being zero, with the most significant rounding bit
    # being 1 but the rest being 0, but with a nonzero remainder after that
    # implying some future bit also being zero, was incorrectly not rounded up.

    # Should not round up.
    print(
        Float64(FloatLiteral(1 << 60) / FloatLiteral(1 << 3))
    )  # CHECK-NEXT: 1.4411518807585587e+17
    # Should round up, but wasn't due to ignoring nonzero remainder in very
    # specific case.
    print(
        Float64(FloatLiteral((1 << 60) + (1 << 7) + 7) / FloatLiteral(1 << 3))
    )  # CHECK-NEXT: 1.441151880758559e+17

    # TODO - with Python semantics this should raise an error, though because
    # they are float literals this would be a static error rather than a dynamic
    # error.
    print(5.0 / 0.0)  # CHECK-NEXT: nan
    print(5.0 / -0.0)  # CHECK-NEXT: nan

    ### Check the combinations of special values for ops
    print("== lhs infinity")  # CHECK-NEXT: lhs infinity
    print(FloatLiteral.infinity + FloatLiteral.infinity)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity + FloatLiteral.negative_infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.infinity + FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity + -0.0)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity + 0)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity + 5)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity - FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity - FloatLiteral.negative_infinity)
    # CHECK-NEXT: inf
    print(FloatLiteral.infinity - FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity - -0.0)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity - 0)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity - 5)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity * FloatLiteral.infinity)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity * FloatLiteral.negative_infinity)
    # CHECK-NEXT: -inf
    print(FloatLiteral.infinity * FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity * -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity * 0)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity * 5)  # CHECK-NEXT: inf
    print(FloatLiteral.infinity / FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity / FloatLiteral.negative_infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.infinity / FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity / -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity / 0)  # CHECK-NEXT: nan
    print(FloatLiteral.infinity / 5)  # CHECK-NEXT: inf

    print("== lhs negative_infinity")  # CHECK-NEXT: lhs negative_infinity
    print(FloatLiteral.negative_infinity + FloatLiteral.infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity + FloatLiteral.negative_infinity)
    # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity + FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity + -0.0)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity + 0)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity + 5)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity - FloatLiteral.infinity)
    # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity - FloatLiteral.negative_infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity - FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity - -0.0)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity - 0)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity - 5)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity * FloatLiteral.infinity)
    # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity * FloatLiteral.negative_infinity)
    # CHECK-NEXT: inf
    print(FloatLiteral.negative_infinity * FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity * -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity * 0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity * 5)  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_infinity / FloatLiteral.infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity / FloatLiteral.negative_infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity / FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity / -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity / 0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_infinity / 5)  # CHECK-NEXT: inf

    print("== lhs negative_zero")  # CHECK-NEXT: lhs negative_zero
    print(FloatLiteral.negative_zero + FloatLiteral.infinity)  # CHECK-NEXT: inf
    print(FloatLiteral.negative_zero + FloatLiteral.negative_infinity)
    # CHECK-NEXT: -inf
    print(FloatLiteral.negative_zero + FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero + -0.0)  # CHECK-NEXT: 0.0
    print(FloatLiteral.negative_zero + 0)  # CHECK-NEXT: 0.0
    print(FloatLiteral.negative_zero + 5)  # CHECK-NEXT: 5
    print(
        FloatLiteral.negative_zero - FloatLiteral.infinity
    )  # CHECK-NEXT: -inf
    print(FloatLiteral.negative_zero - FloatLiteral.negative_infinity)
    # CHECK-NEXT: inf
    print(FloatLiteral.negative_zero - FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero - -0.0)  # CHECK-NEXT: 0.0
    print(FloatLiteral.negative_zero - 0)  # CHECK-NEXT: -0.0
    print(FloatLiteral.negative_zero - 5)  # CHECK-NEXT: -5
    print(FloatLiteral.negative_zero * FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero * FloatLiteral.negative_infinity)
    # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero * FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero * -0.0)  # CHECK-NEXT: 0.0
    print(FloatLiteral.negative_zero * 0)  # CHECK-NEXT: -0.0
    print(FloatLiteral.negative_zero * 5)  # CHECK-NEXT: -0.0
    print(
        FloatLiteral.negative_zero / FloatLiteral.infinity
    )  # CHECK-NEXT: -0.0
    print(FloatLiteral.negative_zero / FloatLiteral.negative_infinity)
    # CHECK-NEXT: 0.0
    print(FloatLiteral.negative_zero / FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero / -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero / 0)  # CHECK-NEXT: nan
    print(FloatLiteral.negative_zero / 5)  # CHECK-NEXT: -0.0

    print("== lhs nan")  # CHECK-NEXT: lhs nan
    print(FloatLiteral.nan + FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan + FloatLiteral.negative_infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan + FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.nan + -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan + 0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan + 5)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - FloatLiteral.negative_infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - 0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan - 5)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * FloatLiteral.negative_infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * 0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan * 5)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / FloatLiteral.negative_infinity)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / FloatLiteral.nan)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / -0.0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / 0)  # CHECK-NEXT: nan
    print(FloatLiteral.nan / 5)  # CHECK-NEXT: nan

    print("== lhs zero")  # CHECK-NEXT: lhs zero
    print(0.0 + FloatLiteral.infinity)  # CHECK-NEXT: inf
    print(0.0 + FloatLiteral.negative_infinity)  # CHECK-NEXT: -inf
    print(0.0 + FloatLiteral.nan)  # CHECK-NEXT: nan
    print(0.0 + -0.0)  # CHECK-NEXT: 0.0
    print(0.0 + 0)  # CHECK-NEXT: 0.0
    print(0.0 + 5)  # CHECK-NEXT: 5
    print(0.0 - FloatLiteral.infinity)  # CHECK-NEXT: -inf
    print(0.0 - FloatLiteral.negative_infinity)  # CHECK-NEXT: inf
    print(0.0 - FloatLiteral.nan)  # CHECK-NEXT: nan
    print(0.0 - -0.0)  # CHECK-NEXT: 0.0
    print(0.0 - 0)  # CHECK-NEXT: 0.0
    print(0.0 - 5)  # CHECK-NEXT: -5
    print(0.0 * FloatLiteral.infinity)  # CHECK-NEXT: nan
    print(0.0 * FloatLiteral.negative_infinity)  # CHECK-NEXT: nan
    print(0.0 * FloatLiteral.nan)  # CHECK-NEXT: nan
    print(0.0 * -0.0)  # CHECK-NEXT: -0.0
    print(0.0 * 0)  # CHECK-NEXT: 0.0
    print(0.0 * 5)  # CHECK-NEXT: 0.0
    print(0.0 / FloatLiteral.infinity)  # CHECK-NEXT: 0.0
    print(0.0 / FloatLiteral.negative_infinity)  # CHECK-NEXT: 0.0
    print(0.0 / FloatLiteral.nan)  # CHECK-NEXT: nan
    print(0.0 / -0.0)  # CHECK-NEXT: nan
    print(0.0 / 0)  # CHECK-NEXT: nan
    print(0.0 / 5)  # CHECK-NEXT: 0.0
