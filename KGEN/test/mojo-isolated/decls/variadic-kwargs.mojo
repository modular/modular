# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Variadic keyword arguments are tested here, because some attributes and types
# need to be checked, and a separate file makes it easier.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-DAG: #[[STRING_TYPE:.*]] = #kgen.parameterizedtype.constant<!String,
# CHECK-DAG: #[[INDEX_TYPE:.*]] = #kgen.concretetype.constant<index,
# CHECK-DAG: #[[MEM_ONLY:.*]] = #kgen.parameterizedtype.constant<!MemOnly,


# CHECK-LABEL: lit.func @"variadic_kwargs
# CHECK-SAME: "[mut [[LT:.*]]](
# CHECK-SAME: %a: index borrow, %b: index borrow, %args: !kgen.variadic<index> borrow|var, *, %c: index borrow, %d: index borrow,
# CHECK-SAME: %kwargs: !lit.ref<{{.*}}@Dict<:!KeyElement #[[STRING_TYPE]], :!CollectionElement #[[INDEX_TYPE]]>, mut [[LT]]> owned_in_mem|var)
fn variadic_kwargs(a: int, b: int, *args: int, c: int, d: int, **kwargs: int):
    pass


fn takes_int_variadic_kwargs(**kwargs: int):
    pass


# CHECK-LABEL: lit.func @"test_variadic_kwargs
fn test_variadic_kwargs():
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: @Dict<:!KeyElement #[[STRING_TYPE]], :!CollectionElement #[[INDEX_TYPE]]>
    # CHECK: lit.call {{.*}}@Dict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[X:.*]] = lit.var.decl {{.*}}!String,
    # CHECK: %[[X_LIT:.*]] = kgen.param.constant: !StringLiteral = <{:string "x"}>
    # CHECK: lit.call {{.*}}@String::@"__init__{{.*}}(%[[X]], %[[X_LIT]])
    # CHECK: %[[X_KEY:.*]] = lit.ref.immut %[[X]]
    # CHECK: %[[X_VAL:.*]] = lit.var.decl {{.*}}index,
    # CHECK: %[[IDX9:.*]] = kgen.param.constant = <9>
    # CHECK: lit.ref.store %[[IDX9]], %[[X_VAL]]
    # CHECK: %[[X_PASSED:.*]] = lit.ref.immut %[[X_VAL]]
    # CHECK: lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%[[DICT_VAR]], %[[X_KEY]], %[[X_PASSED]])

    # CHECK: %[[S:.*]] = lit.var.decl {{.*}}!String,
    # CHECK: %[[S_LIT:.*]] = kgen.param.constant: !StringLiteral = <{:string "stuff"}>
    # CHECK: lit.call {{.*}}@String::@"__init__{{.*}}(%[[S]], %[[S_LIT]])
    # CHECK: %[[S_KEY:.*]] = lit.ref.immut %[[S]]
    # CHECK: %[[S_VAL:.*]] = lit.var.decl {{.*}}index,
    # CHECK: %[[IDX8:.*]] = kgen.param.constant = <8>
    # CHECK: lit.ref.store %[[IDX8]], %[[S_VAL]]
    # CHECK: %[[S_PASSED:.*]] = lit.ref.immut %[[S_VAL]]
    # CHECK: lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%[[DICT_VAR]], %[[S_KEY]], %[[S_PASSED]])

    # CHECK lit.call {{.*}}@"takes_int_variadic_kwargs{{.*}}(%[[DICT_VAR]])
    takes_int_variadic_kwargs(x=`9`, stuff=`8`)


trait SomeTrait(CollectionElement):
    pass


fn infers_param_from_kwargs[T: SomeTrait](**kwargs: T):
    pass


@value
struct MemOnly(SomeTrait):
    pass


# CHECK-LABEL: lit.func @"test_variadic_kwargs_param_inference
fn test_variadic_kwargs_param_inference():
    # %s = lit.var.decl "s" var : !lit.ref<!MemOnly,
    var s = MemOnly()

    # CHECK: %[[M:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!MemOnly,
    # CHECK-NEXT: lit.call {{.*}}@MemOnly::@"__init__{{.*}}(%[[M]])

    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: @Dict<:!KeyElement #[[STRING_TYPE]], :!CollectionElement #[[MEM_ONLY]]>
    # CHECK: lit.call {{.*}}@Dict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[Y:.*]] = lit.var.decl {{.*}}!String,
    # CHECK: %[[Y_LIT:.*]] = kgen.param.constant: !StringLiteral = <{:string "y"}>
    # CHECK: lit.call {{.*}}@String::@"__init__{{.*}}(%[[Y]], %[[Y_LIT]])
    # CHECK: %[[Y_KEY:.*]] = lit.ref.immut %[[Y]]
    # CHECK: %[[Y_PASSED:.*]] = lit.ref.immut %[[M]] : <!MemOnly
    # CHECK: lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%[[DICT_VAR]], %[[Y_KEY]], %[[Y_PASSED]])

    # CHECK: %[[Z:.*]] = lit.var.decl {{.*}}!String
    # CHECK: %[[Z_LIT:.*]] = kgen.param.constant: !StringLiteral = <{:string "z"}>
    # CHECK: lit.call {{.*}}@String::@"__init__{{.*}}(%[[Z]], %[[Z_LIT]])
    # CHECK: %[[Z_KEY:.*]] = lit.ref.immut %[[Z]]
    # CHECK: %[[S_PASSED:.*]] = lit.ref.immut %s
    # CHECK: lit.call {{.*}}@Dict::@"__setitem__{{.*}}(%[[DICT_VAR]], %[[Z_KEY]], %[[S_PASSED]])
    infers_param_from_kwargs(y=MemOnly(), z=s)


# COM: test that the inferred type of variables is correct when the initializer
# COM: expression has variadic keyword arguments.
# COM: Issue https://github.com/modularml/modular/issues/35215
fn takes_kw(**kwargs: MemOnly) -> int:
    return `0`


# CHECK-LABEL: lit.func @"test_takes_kw_in_assignment
fn test_takes_kw_in_assignment(x: MemOnly):
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<index,
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: @Dict<:!KeyElement #[[STRING_TYPE]], :!CollectionElement #[[MEM_ONLY]]>
    # CHECK: %[[RES:.*]] = lit.call {{.*}}@"takes_kw{{.*}}(%[[DICT_VAR]])
    # CHECK: lit.ref.store %[[RES]], %b
    var b = takes_kw(y=x, z=x)
