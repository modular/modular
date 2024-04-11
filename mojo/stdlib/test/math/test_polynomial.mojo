# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math.polynomial import (
    EvaluationMethod,
    _estrin_evaluate,
    _horner_evaluate,
    polynomial_evaluate,
)


# CHECK-LABEL: test_polynomial_evaluate_degree3
fn test_polynomial_evaluate_degree3():
    print("== test_polynomial_evaluate_degree3")

    alias simd_width = 1
    alias coeefs = List[SIMD[DType.float64, simd_width]](
        1000.0,
        1.0,
        1.0,
    )
    # Evaluate 1000 + x + x^2
    var y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1002.0
    print(y)

    y = _estrin_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1002.0
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1002.0
    print(y)

    y = polynomial_evaluate[
        DType.float64, simd_width, coeefs, method = EvaluationMethod.ESTRIN
    ](1.0)

    # CHECK: 1002.0
    print(y)

    y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1000.11
    print(y)

    y = _estrin_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1000.11
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1000.11
    print(y)


# CHECK-LABEL: test_polynomial_evaluate_degree4
fn test_polynomial_evaluate_degree4():
    print("== test_polynomial_evaluate_degree4")

    alias simd_width = 1
    alias coeefs = List[SIMD[DType.float64, simd_width]](
        1000.0,
        99.0,
        -43.0,
        12.0,
        -14.0,
    )
    # Evalaute 1000 + 99 x - 43 x^2 + 12 x^3 - 14 x^4
    var y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1054.0
    print(y)

    y = _estrin_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1054.0
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1054.0
    print(y)

    y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1009.4806
    print(y)

    y = _estrin_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1009.4806
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1009.4806
    print(y)


# COM: Note that the estrin method currently goes up to degree 9
# CHECK-LABEL: test_polynomial_evaluate_degree10
fn test_polynomial_evaluate_degree10():
    print("== test_polynomial_evaluate_degree10")

    alias simd_width = 1
    alias coeefs = List[SIMD[DType.float64, simd_width]](
        20.0,
        9.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        43.0,
        10.0,
    )
    # Evaluate
    # 20.0 + 9.0 x + 1.0 x^2 + 1.0 x^3 + 1.0 x^4 + 1.0 x^5 + 1.0 x^6 +
    # 1.0 x^7 + 1.0 x^8 + 43.0 x^9 + 10.0 x^10
    var y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 89.0
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 89.0
    print(y)

    y = _horner_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 20.91{{[0-9]+}}
    print(y)

    y = polynomial_evaluate[
        DType.float64,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 20.91{{[0-9]+}}
    print(y)


fn main():
    test_polynomial_evaluate_degree3()
    test_polynomial_evaluate_degree4()
    test_polynomial_evaluate_degree10()
