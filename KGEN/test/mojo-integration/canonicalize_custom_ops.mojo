# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate -S %s | FileCheck %s

import _mlir
from _mlir import (
    Operation,
    Rewriter,
    Value,
    Type,
    Location,
    NamedAttribute,
    Attribute,
    Identifier,
)
from compile.reflection import get_linkage_name
from collections import Optional


fn Op[
    T: AnyTrivialRegType, //, func: T
](
    loc: Location,
    operands: List[Value],
    results: List[Type],
    params: Optional[Attribute] = None,
) -> Operation:
    alias name: StringLiteral = "custom." + get_linkage_name[func]()
    var attrs = List[NamedAttribute]()
    if params:
        attrs.append(
            NamedAttribute(
                Identifier(loc.context(), "params"), params.unsafe_value()
            )
        )
    return Operation(
        name, loc, operands=operands, results=results, attributes=attrs
    )


@op
fn mul_two[T: AnyType](x: Int32) -> Int32:
    return x * 2


@op(add_mul_two[1])
fn add[T: AnyType](x: Int32, y: Int32) -> Int32:
    return x + y


fn add_mul_two[x: Int](inout op: Operation, inout b: Rewriter) -> Bool:
    var loc = op.location()
    var lhs = op.operand(0)
    var rhs = op.operand(1)
    if lhs != rhs:
        return True

    var new_op = Op[mul_two](
        loc,
        operands=List[Value](lhs),
        results=List[Type](op.result(0).type()),
        params=op.get_discardable_attr("params"),
    )
    _ = b.insert(new_op)
    b.replace_op_with(op, new_op)
    return True


# CHECK-LABEL: kgen.func export @top
@export
fn top(x: Int32) -> Int32:
    # CHECK-NEXT: call @__op_inst_0(%arg0)
    return add[Int32](x, x)


# CHECK-LABEL: kgen.func @__op_inst_0
# CHECK-NEXT: call [[CALLEE:.*mul_two.*,T=.*"]](%arg0)

# CHECK: kgen.func [[CALLEE]]
# CHECK-NEXT: %simd = kgen.param.constant: scalar<si32> = <2>
# CHECK-NEXT: pop.mul %arg0, %simd
