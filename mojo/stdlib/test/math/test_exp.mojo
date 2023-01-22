# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_exp::main():index()' -I %stdlibdir | FileCheck %s

from DType import DType
from IO import print
from Math import exp
from SIMD import SIMD


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp\n")

    # CHECK: 7.389056
    print(exp[1, DType.f32.value](SIMD[1, DType.f32.value](2)))


@export
fn main() -> __mlir_type.index:
    test_exp()
    return 0
