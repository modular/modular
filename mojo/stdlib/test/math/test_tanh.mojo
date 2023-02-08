# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from DType import DType
from Int import Int
from IO import print
from Math import tanh, iota


# CHECK-LABEL: test_tanh
fn test_tanh():
    print("== test_tanh\n")

    let simd_val = 0.5 * iota[4, DType.f32.value]()

    # CHECK: [0.000000, 0.462117, 0.761594, 0.905148]
    print(tanh(simd_val))

    # CHECK: [0.000000, 0.244919, 0.462117, 0.635149]
    print(tanh(0.5 * simd_val))


fn main():
    test_tanh()
