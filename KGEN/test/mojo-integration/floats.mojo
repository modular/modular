# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn main():
    print(Float64(1.1))  # CHECK: 1.10000{{.*}}
    print(Float64(0.1))  # CHECK: 0.10000{{.*}}
    print(Float64(1.0))  # CHECK: 1
    print(Float64(1e2))  # CHECK: 100
    print(Float64(1.1e2))  # CHECK: 110
    print(Float64(0.1e2))  # CHECK: 10
    print(Float64(1.0e2))  # CHECK: 100
    print(Float64(1e2))  # CHECK: 100
    print(Float64(1.1e-2))  # CHECK: 0.01099{{.*}}
    print(Float64(0.1e2))  # CHECK: 10
    print(Float64(1.0e-2))  # CHECK: 0.01
    print(Float64(0.1))  # CHECK: 0.100000{{.*}}
    print(Float64(0.0))  # CHECK: 0
    print(Float64(0e2))  # CHECK: 0
    print(Float64(0.1e2))  # CHECK: 10
    print(Float64(0.0e2))  # CHECK: 0
    print(Float64(0e2))  # CHECK: 0
    print(Float64(0.1e-2))  # CHECK: 0.001
    print(Float64(0.0e-2))  # CHECK: 0
    print(Float64(12.31e11))  # CHECK: 1231000000000.0
    print(Float64(12.31e-3))  # CHECK: 0.01231
    # Check gradual loss of precision for subnormal numbers when
    # converting from infinite precision literal to Float64.
    print(Float64(1.1234567e-305))  # CHECK: 1.1234567e-305
    print(Float64(1.1234567e-310))  # CHECK: 1.1234567000000234e-310
    print(Float64(1.1234567e-315))  # CHECK: 1.1234567021086955e-315
    print(Float64(1.1234567e-320))  # CHECK: 1.1235052786429946e-320
    print(Float64(1.1234567e-322))  # CHECK: 1.1363509854348671e-322
    print(Float64(1.1234567e-323))  # CHECK: 9.8813129168249309e-324
    print(Float64(1.1234567e-324))  # CHECK: 0
