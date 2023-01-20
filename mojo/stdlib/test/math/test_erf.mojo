# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_erf::main():index()' -I %stdlibdir | FileCheck %s


from IO import print
from Math import erf
from SIMD import SIMD


# CHECK-LABEL: test_erf
fn test_erf():
    print("== test_erf\n")

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(
        erf[2, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
            SIMD[2, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](2)
        )
    )


@export
fn main() -> __mlir_type.index:
    test_erf()
    return 0
