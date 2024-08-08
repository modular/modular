# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | kgen-opt -lower-semantic-cf --check-lifetimes -lower-lit -lower-lit-types -register-custom-ops -outline-closures -lift-and-fold-apply -elaborate-generators -lower-custom-ops | FileCheck %s

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


# CHECK: %[[PARAM:.*]] = kgen.param.constant: scalar<si32> = <12>
# CHECK: %{{.*}} = pop.add %{{.*}}, %[[PARAM:.*]] : !pop.scalar<si32>

# CHECK: %[[PARAM2:.*]] = kgen.param.constant: scalar<si32> = <30>
# CHECK: %{{.*}} = pop.add %{{.*}}, %[[PARAM2:.*]] : !pop.scalar<si32>
