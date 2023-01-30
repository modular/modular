# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_erf::main():index()' -I %stdlibdir | FileCheck %s


from DType import DType
from IO import print
from Math import erf
from SIMD import SIMD


# CHECK-LABEL: test_erf
fn test_erf():
    print("== test_erf\n")

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[2, DType.f32.value](2)))


@export
fn main() -> __mlir_type.index:
    test_erf()
    return 0
