# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import frexp, isinf, log, log2


# CHECK-LABEL: test_frexp
fn test_frexp():
    print("== test_frexp")

    # CHECK: 0.964453
    # CHECL: 7.0
    var res = frexp(Float32(123.45))
    print(res[0])
    print(res[1])

    # CHECK: 0.8
    # CHECK: -3.0
    res = frexp(Float32(0.1))
    print(res[0])
    print(res[1])

    # CHECK: -0.8
    # CHECK: -3.0
    res = frexp(Float32(-0.1))
    print(res[0])
    print(res[1])

    # CHECK: [0.0, 0.5, 0.5, 0.625]
    # CHECK: [0.0, 2.0, 3.0, 3.0]
    var res2 = frexp(SIMD[DType.float32, 4](0, 2, 4, 5))
    print(res2[0])
    print(res2[1])


# CHECK-LABEL: test_log
fn test_log():
    print("== test_log")

    # CHECK: 4.8158{{[0-9]+}}
    print(log(Float32(123.45)))

    # CHECK: -2.3025{{[0-9]+}}
    print(log(Float32(0.1)))

    # CHECK: [0.0, 0.693147{{[0-9]+}}, 1.38629{{[0-9]+}}, 1.6094{{[0-9]+}}]
    print(log(SIMD[DType.float32, 4](1, 2, 4, 5)))

    # CHECK: 1.0
    print(log[DType.float32, 1](2.7182818284590452353602874713526624977572))

    # CHECK: [True, False, True, True]
    print(isinf(log(SIMD[DType.float32, 4](0, 1, 0, 0))))


# CHECK-LABEL: test_log2
fn test_log2():
    print("== test_log2")

    # CHECK: 6.9477{{[0-9]+}}
    print(log2(Float32(123.45)))

    # CHECK: -3.3219{{[0-9]+}}
    print(log2(Float32(0.1)))

    # CHECK: [0.0, 1.0, 2.0, 2.321{{[0-9]+}}]
    print(log2(SIMD[DType.float32, 4](1, 2, 4, 5)))


fn main():
    test_frexp()
    test_log()
    test_log2()
