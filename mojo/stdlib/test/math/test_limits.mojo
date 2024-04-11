# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math.limit import max_finite, min_finite


# CHECK-LABEL: test_numeric_limits
fn test_numeric_limits():
    print("== test_numeric_limits")

    @parameter
    fn overflow_int[type: DType]():
        constrained[
            type.is_integral(), "comparison only valid on integral types"
        ]()
        print(max_finite[type]() + 1 < max_finite[type]())

    @parameter
    fn overflow_fp[type: DType]():
        constrained[
            type.is_floating_point(),
            "comparison only valid on floating point types",
        ]()
        print(max_finite[type]() + 1 == max_finite[type]())

    @parameter
    fn underflow_int[type: DType]():
        constrained[
            type.is_integral(), "comparison only valid on integral types"
        ]()
        print(min_finite[type]() - 1 > min_finite[type]())

    @parameter
    fn underflow_fp[type: DType]():
        constrained[
            type.is_floating_point(),
            "comparison only valid on floating point types",
        ]()
        print(min_finite[type]() - 1 == min_finite[type]())

    # CHECK: True
    overflow_int[DType.int8]()
    # CHECK: True
    overflow_int[DType.uint8]()
    # CHECK: True
    overflow_int[DType.int16]()
    # CHECK: True
    overflow_int[DType.uint16]()
    # CHECK: True
    overflow_int[DType.int32]()
    # CHECK: True
    overflow_int[DType.uint32]()
    # CHECK: True
    overflow_int[DType.int64]()
    # CHECK: True
    overflow_int[DType.uint64]()

    # CHECK: True
    overflow_fp[DType.float32]()
    # CHECK: True
    overflow_fp[DType.float64]()

    # CHECK: True
    underflow_int[DType.int8]()
    # CHECK: True
    underflow_int[DType.uint8]()
    # CHECK: True
    underflow_int[DType.int16]()
    # CHECK: True
    underflow_int[DType.uint16]()
    # CHECK: True
    underflow_int[DType.int32]()
    # CHECK: True
    underflow_int[DType.uint32]()
    # CHECK: True
    underflow_int[DType.int64]()
    # CHECK: True
    underflow_int[DType.uint64]()

    # CHECK: True
    underflow_fp[DType.float32]()
    # CHECK: True
    underflow_fp[DType.float64]()


fn main():
    test_numeric_limits()
