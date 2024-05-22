# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo %s | FileCheck %s

fn mogg_register(x: StringLiteral):
    pass


struct MemoryType:
    pass


struct ParamType[a: Int]:
    pass

# CHECK: [[PARAM:#.*]] = #kgen.parameterizedtype.constant<!lit.declref<#ParamType <:!Int a>

# CHECK: lit.func @"custom_op_args
# CHECK-NEXT: mogg.arg_conformances = [#kgen<exprs[{{.*}}]>, #kgen<exprs[{{.*}}]>]
# CHECK-SAME: mogg.result_conformances = #kgen<exprs[{{.*}}]>
@mogg_register("custom.op")
fn custom_op_args(a: Int, b: MemoryType):
    pass


@mogg_register("custom.op")
fn custom_op_varargs(*a: Int, **b: Int) raises -> Int:
    pass


@mogg_register("custom.op")
fn custom_op_generic[T: Movable, *Ts: Copyable](a: T, *b: *Ts) -> MemoryType:
    pass


# CHECK: lit.func @"custom_op_param
# CHECK-NEXT: mogg.arg_conformances = [#kgen<exprs[[[PARAM]]]>,
# CHECK-SAME: mogg.result_conformances = #kgen<exprs[[[PARAM]]]>
@mogg_register("custom.op")
fn custom_op_param[a: Int](b: ParamType[a], c: ParamType[1]) -> ParamType[a]:
    pass
