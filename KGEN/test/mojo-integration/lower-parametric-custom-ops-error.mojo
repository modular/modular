# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | kgen-opt -lower-semantic-cf --check-lifetimes -lower-lit -lower-lit-types -register-custom-ops -outline-closures -lift-and-fold-apply -elaborate-generators -lower-custom-ops 2>&1 | FileCheck %s

import _mlir
from _mlir import Operation, Rewriter, Value, Type


@op_implementation("custom.add_constant")
struct AddConstantOp:
    @staticmethod
    fn impl[type: Int32](x: Int32) -> Int32:
        return x + type


fn main():
    var x: Int32 = 30
    # CHECK: no implementation found for custom op 'custom.unknown_op'
    var res = __mlir_op.`custom.unknown_op`[
        _type=Int32, _op_impl_params = Int32(12)
    ](x)
    print(res)
