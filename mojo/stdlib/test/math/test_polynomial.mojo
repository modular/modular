# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from DType import DType
from Polynomial import polynomial_evaluate, _horner_evaluate
from IO import print
from List import VariadicList


# CHECK-LABEL: test_polynomial_evaluate_degree3
fn test_polynomial_evaluate_degree3():
    print("== test_polynomial_evaluate_degree3\n")

    alias simd_width = 1
    alias coeefs = VariadicList[__mlir_type.f64](
        1000.0,
        1.0,
        1.0,
    )
    # Evaluate 1000 + x + x^2
    var y = _horner_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1002.000000
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1002.000000
    print(y)

    y = _horner_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1000.110000
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1000.110000
    print(y)


# CHECK-LABEL: test_polynomial_evaluate_degree4
fn test_polynomial_evaluate_degree4():
    print("== test_polynomial_evaluate_degree4\n")

    alias simd_width = 1
    alias coeefs = VariadicList[__mlir_type.f64](
        1000.0,
        99.0,
        -43.0,
        12.0,
        -14.0,
    )
    # Evalaute 1000 + 99 x - 43 x^2 + 12 x^3 - 14 x^4
    var y = _horner_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1054.000000
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 1054.000000
    print(y)

    y = _horner_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1009.480600
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 1009.480600
    print(y)


# CHECK-LABEL: test_polynomial_evaluate_degree10
fn test_polynomial_evaluate_degree10():
    print("== test_polynomial_evaluate_degree10\n")

    alias simd_width = 1
    alias coeefs = VariadicList[__mlir_type.f64](
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
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 89.000000
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](1.0)

    # CHECK: 89.000000
    print(y)

    y = _horner_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 20.911111
    print(y)

    y = polynomial_evaluate[
        __mlir_type.f64,
        DType.f64.value,
        simd_width,
        coeefs,
    ](0.1)

    # CHECK: 20.911111
    print(y)


fn main():
    test_polynomial_evaluate_degree3()
    test_polynomial_evaluate_degree4()
    test_polynomial_evaluate_degree10()
