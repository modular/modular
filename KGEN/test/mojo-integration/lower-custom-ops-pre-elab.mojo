# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | kgen-opt -lower-semantic-cf --check-lifetimes -lower-lit -lower-lit-types -lower-custom-ops-pre-elab | FileCheck %s

import _mlir


# Parametric op implementation
@op_implementation("custom.add_constant")
struct AddConstantOp:
    @staticmethod
    fn impl[cst: Int32](x: Int32) -> Int32:
        return x + cst


# Non-parametric op implementation
@op_implementation("custom.add_thirty")
struct AddThirtyOp:
    @staticmethod
    fn impl(x: Int32) -> Int32:
        return x + 30


fn main():
    var x: Int32 = 30
    var res = __mlir_op.`custom.add_constant`[
        _type=Int32, _op_impl_params = Int32(12)
    ](x)
    print("The answer is:", res)

    var y: Int32 = 12
    var res2 = __mlir_op.`custom.add_thirty`[_type=Int32](y)
    print("The answer is still:", res2)


# CHECK-NOT: custom.add_constant
# CHECK: kgen.call @"{{.*}}AddConstantOp::impl{{.*}}"<:scalar<si32>

# CHECK-NOT: custom.add_thirty
# CHECK: kgen.call @"{{.*}}AddThirtyOp::impl{{.*}}"
