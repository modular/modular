# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_tanh::main():index()' -I %stdlibdir | FileCheck %s

from IO import print
from Math import tanh, iota


# CHECK-LABEL: test_tanh
fn test_tanh():
    print("== test_tanh\n")

    let simd_val = (
        0.5 * iota[4, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`]()
    )

    # CHECK: [0.000000, 0.462117, 0.761594, 0.905148]
    print[4, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
        tanh[4, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](simd_val)
    )

    # CHECK: [0.000000, 0.244919, 0.462117, 0.635149]
    print[4, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
        tanh[4, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
            0.5 * simd_val
        )
    )


@export
fn main() -> __mlir_type.index:
    test_tanh()
    return 0
