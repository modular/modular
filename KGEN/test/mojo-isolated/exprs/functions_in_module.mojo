# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

# CHECK-DAG: [[TYPE1:#.*]] = #kgen.type<{{.*}}#MLIRType <:type !lit.generator<() -> !kgen.none>>{{.*}} : !AnyType
# CHECK-DAG: [[TYPE2:#.*]] = #kgen.type<{{.*}}#MLIRType <:type !lit.generator<("x": !Int) -> !kgen.none>>{{.*}} : !AnyType
# CHECK-DAG: [[TYPE3:#.*]] = #kgen.type<{{.*}}#MLIRType <:type !lit.generator<("y": !Int, "z": !Int) -> !kgen.none>>{{.*}} : !AnyType
# CHECK-DAG: [[TYPE4:#.*]] = #kgen.type<{{.*}}#MLIRType <:type !lit.generator<[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1>>{{.*}} : !AnyType
# CHECK-DAG: [[TYPE5:#.*]] = #kgen.type<{{.*}}#MLIRType <:type !lit.generator<<"func_type": !AnyType, +, "func": !kgen.param<:!AnyType *(0,0)>>() -> !kgen.none>>{{.*}} : !AnyType

fn foo(): pass

fn bar(x: Int): pass

# NOTE: this is intentionally in the middle here, to ensure that the intrinsic
# correctly resolves signatures that are declared after the call.

# CHECK: lit.alias.decl {{.*}}#Tuple <:variadic<!AnyType> [[[TYPE1]], [[TYPE2]], [[TYPE3]], [[TYPE4]], [[TYPE5]]]>
# CHECK-SAME: <store_to_mem(@functions_in_module::@"foo()"), store_to_mem(@functions_in_module::@"bar(::Int)"), store_to_mem(@functions_in_module::@"bar(::Int,::Int)"), store_to_mem(@functions_in_module::@"baz()"), store_to_mem(@functions_in_module::@"take[::AnyType,$0]()")>))))>
comptime funcs = __functions_in_module()

# CHECK-LABEL: lit.fn @"main
fn main():
    # CHECK-NEXT: lit.call @functions_in_module::@"take[::AnyType,$0]()"<:!AnyType [[TYPE1]],
    take[funcs[0]]()
    # CHECK-NEXT: lit.call @functions_in_module::@"take[::AnyType,$0]()"<:!AnyType [[TYPE2]],
    take[funcs[1]]()
    # CHECK-NEXT: lit.call @functions_in_module::@"take[::AnyType,$0]()"<:!AnyType [[TYPE3]],
    take[funcs[2]]()
    # CHECK-NEXT: lit.call @functions_in_module::@"take[::AnyType,$0]()"<:!AnyType [[TYPE4]],
    take[funcs[3]]()
    # CHECK-NEXT: lit.call @functions_in_module::@"take[::AnyType,$0]()"<:!AnyType [[TYPE5]],
    take[funcs[4]]()

fn bar(y: Int, z: Int): pass

fn baz() raises: pass

fn take[func_type: AnyType, //, func: func_type]():
    pass
