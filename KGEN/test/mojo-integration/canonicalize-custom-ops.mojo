# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | kgen-opt -lower-semantic-cf --check-lifetimes -lower-lit -lower-lit-types -register-custom-ops -cse -canonicalizer | FileCheck %s

import _mlir
from _mlir import Operation, Rewriter, Value, Type


@op_implementation("custom.mul_two")
struct MulTwoOp:
    @staticmethod
    fn impl(x: Int32) -> Int32:
        return x * 2


@op_implementation("custom.add")
struct AddOp:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

    @staticmethod
    fn canonicalize(inout op: Operation, inout rewriter: Rewriter) -> Bool:
        var loc = op.location()
        var lhs = op.operand(0)
        var rhs = op.operand(1)
        if lhs != rhs:
            return True

        var new_op = Operation(
            "custom.mul_two",
            loc,
            operands=List[Value](lhs),
            results=List[Type](op.result(0).type()),
        )
        _ = rewriter.insert(new_op)
        rewriter.replace_op_with(op, new_op)
        return True


fn main():
    var x: Int32 = 4
    var res = __mlir_op.`custom.add`[_type=Int32](x, x)
    print(res)


# CHECK-NOT: %{{.*}} = "custom.add"(%{{.*}})
# CHECK: %{{.*}} = "custom.mul_two"(%{{.*}})
