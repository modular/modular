# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo %s | FileCheck %s


fn register_internal(x: StaticString):
    pass


struct MemoryType:
    pass


struct ParamType[a: Int]:
    pass


# CHECK: lit.fn @"custom_op_args
# CHECK-NEXT: mogg.arg_value_witnesses = [{__del__ = {{[^}]*}}, __moveinit__ = {{[^\}]*}}}, {__del__ = {{[^}]*}}}]
# CHECK-SAME: mogg.result_value_witnesses = {__del__ = {{[^}]*}}}
@register_internal("custom.op")
fn custom_op_args(a: Int, b: MemoryType):
    pass


@register_internal("custom.op")
fn custom_op_varargs(*a: Int, **b: Int) raises -> Int:
    pass


@register_internal("custom.op")
fn custom_op_generic[T: Movable, *Ts: Copyable](a: T, *b: *Ts) -> MemoryType:
    pass


# CHECK: lit.fn @"custom_op_param
# CHECK-NEXT: mogg.arg_value_witnesses = [{__del__ = #kgen.get_witness<#kgen.type<@mogg::@ParamType<:!Int a>> : {{[^}]*}}},
# CHECK-SAME: mogg.result_value_witnesses = {__del__ = #kgen.get_witness<#kgen.type<@mogg::@ParamType<:!Int a>> : {{[^}]*}}}
@register_internal("custom.op")
fn custom_op_param[a: Int](b: ParamType[a], c: ParamType[1]) -> ParamType[a]:
    pass
