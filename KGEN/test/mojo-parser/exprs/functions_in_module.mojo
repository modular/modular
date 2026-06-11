# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

# CHECK-DAG: [[TYPE1:#.*]] = #kgen.type<{{.*}}#MLIRType <:non_struct_type !lit.generator<() -> !kgen.none>>{{.*}} : !Movable
# CHECK-DAG: [[TYPE2:#.*]] = #kgen.type<{{.*}}#MLIRType <:non_struct_type !lit.generator<("x": !Int) -> !kgen.none>>{{.*}} : !Movable
# CHECK-DAG: [[TYPE3:#.*]] = #kgen.type<{{.*}}#MLIRType <:non_struct_type !lit.generator<("y": !Int, "z": !Int) -> !kgen.none>>{{.*}} : !Movable
# CHECK-DAG: [[TYPE4:#.*]] = #kgen.type<{{.*}}#MLIRType <:non_struct_type !lit.generator<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>>{{.*}} : !Movable
# CHECK-DAG: [[TYPE5:#.*]] = #kgen.type<{{.*}}#MLIRType <:non_struct_type !lit.generator<<"func_type": !Movable, +, "func": !kgen.param<:!Movable *(0,0)>>() -> !kgen.none>>{{.*}} : !Movable


def foo():
    pass


def bar(x: Int):
    pass


# NOTE: this is intentionally in the middle here, to ensure that the intrinsic
# correctly resolves signatures that are declared after the call.

# CHECK: lit.alias.decl {{.*}}#Tuple <:param_list<!Movable> [[[TYPE1]], [[TYPE2]], [[TYPE3]], [[TYPE4]], [[TYPE5]]]
# CHECK-SAME: <store_to_mem(@functions_in_module::@"foo()"), store_to_mem(@functions_in_module::@"bar(::Int)"), store_to_mem(@functions_in_module::@"bar(::Int,::Int)"), store_to_mem(@functions_in_module::@"baz()"), store_to_mem(@functions_in_module::@"take[::Movable,$0]()"{{.*}})>))
comptime funcs = __functions_in_module()


# CHECK-LABEL: lit.fn @"main
def main():
    # CHECK-NEXT: lit.call {{.*}}@"take[::Movable,$0]()"<:!Movable [[TYPE1]],
    take[funcs[0]]()
    # CHECK-NEXT: lit.call {{.*}}@"take[::Movable,$0]()"<:!Movable [[TYPE2]],
    take[funcs[1]]()
    # CHECK-NEXT: lit.call {{.*}}@"take[::Movable,$0]()"<:!Movable [[TYPE3]],
    take[funcs[2]]()
    # CHECK-NEXT: lit.call {{.*}}@"take[::Movable,$0]()"<:!Movable [[TYPE4]],
    take[funcs[3]]()
    # CHECK-NEXT: lit.call {{.*}}@"take[::Movable,$0]()"<:!Movable [[TYPE5]],
    take[funcs[4]]()


def bar(y: Int, z: Int):
    pass


def baz() raises:
    pass


def take[func_type: Movable, //, func: func_type]():
    pass
