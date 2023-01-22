# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_tanh::main():index()' -I %stdlibdir | FileCheck %s

from DType import DType
from IO import print
from Math import tanh, iota


# CHECK-LABEL: test_tanh
fn test_tanh():
    print("== test_tanh\n")

    let simd_val = 0.5 * iota[4, DType.f32.value]()

    # CHECK: [0.000000, 0.462117, 0.761594, 0.905148]
    print[4, DType.f32.value](tanh[4, DType.f32.value](simd_val))

    # CHECK: [0.000000, 0.244919, 0.462117, 0.635149]
    print[4, DType.f32.value](tanh[4, DType.f32.value](0.5 * simd_val))


@export
fn main() -> __mlir_type.index:
    test_tanh()
    return 0
