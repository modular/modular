# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


def register_internal(x: StaticString):
    pass


struct MemoryType:
    pass


struct ParamType[a: Int]:
    pass


# CHECK: lit.fn @"custom_op_args
# CHECK-NEXT: mogg.arg_value_witnesses = [{__del__{{.*}} = {{[^}]*}},{{.*}}__init__ = {{.*}}"__init__(take:::Int$)"{{.*}}, {__del__{{.*}} = {{[^}]*}}}]
# CHECK-SAME: mogg.result_value_witnesses = {__del__{{.*}} = {{[^}]*}}}
@register_internal("custom.op")
def custom_op_args(a: Int, b: MemoryType):
    pass


@register_internal("custom.op")
def custom_op_varargs(*a: Int, **b: Int) raises -> Int:
    pass


@register_internal("custom.op")
def custom_op_generic[
    T: Movable, *Ts: ImplicitlyCopyable
](a: T, *b: *Ts) -> MemoryType:
    pass


# CHECK: lit.fn @"custom_op_param
# CHECK-NEXT: mogg.arg_value_witnesses = [{__del__{{.*}} = #kgen.symbol.constant<@mogg::@ParamType::@"__del__(mogg::ParamType[$0]$)"<:!Int a>>{{.*}}}, {__del__{{.*}} = #kgen.symbol.constant<@mogg::@ParamType::@"__del__(mogg::ParamType[$0]$)"<:!Int {1}>>{{.*}}}]
# CHECK-SAME: mogg.result_value_witnesses = {__del__{{.*}} = #kgen.symbol.constant<@mogg::@ParamType::@"__del__(mogg::ParamType[$0]$)"<:!Int a>>{{.*}}}
@register_internal("custom.op")
def custom_op_param[a: Int](b: ParamType[a], c: ParamType[1]) -> ParamType[a]:
    pass


# CHECK: lit.fn @"unknown_type
# CHECK-NEXT: mogg.arg_value_witnesses = [{__init__ = {{.*}}"__init__(take:$0$)"{{.*}}mogg.result_value_witnesses = {__init__ = {{.*}}"__init__(take:$0$)"
@register_internal("custom.op")
def unknown_type[T: Movable](a: T) -> T:
    pass
