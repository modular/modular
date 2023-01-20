# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_exp::main():index()' -I %stdlibdir | FileCheck %s

from SIMD import SIMD
from IO import print
from Math import exp


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp\n")

    # CHECK: 7.389056
    print(
        exp[1, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
            SIMD[1, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](2)
        )
    )


@export
fn main() -> __mlir_type.index:
    test_exp()
    return 0
