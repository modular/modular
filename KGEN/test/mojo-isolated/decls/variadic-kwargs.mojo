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

# CHECK-DAG: #[[INDEX_TYPE:.*]] = #kgen.type<index,
# CHECK-DAG: #[[MEM_ONLY:.*]] = #kgen.type<!MemOnly,


# CHECK-LABEL: lit.func @"variadic_kwargs
# CHECK-SAME: "[mut [[LT:.*]]](
# CHECK-SAME: %a: index, %b: index, %args: !kgen.variadic<index> var, *, %c: index, %d: index,
# CHECK-SAME: %kwargs: !lit.ref<{{.*}}@OwnedKwargsDict<:!CollectionElement #[[INDEX_TYPE]]>, mut [[LT]]> owned_in_mem|var)
fn variadic_kwargs(a: int, b: int, *args: int, c: int, d: int, **kwargs: int):
    pass


fn takes_int_variadic_kwargs(**kwargs: int):
    pass


fn takes_int_variadic_kwargs_multiline(
    **kwargs: int,
):
    pass


# CHECK-LABEL: lit.func @"test_variadic_kwargs
fn test_variadic_kwargs():
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: @OwnedKwargsDict<:!CollectionElement #[[INDEX_TYPE]]>
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[X_KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "x"}>
    # CHECK: %[[X_VAL:.*]] = lit.var.decl {{.*}}index,
    # CHECK: %[[IDX9:.*]] = kgen.param.constant = <9>
    # CHECK: lit.ref.store %[[IDX9]], %[[X_VAL]]
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[X_KEY]], %[[X_VAL]])

    # CHECK: %[[S_KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "stuff"}>
    # CHECK: %[[S_VAL:.*]] = lit.var.decl {{.*}}index,
    # CHECK: %[[IDX8:.*]] = kgen.param.constant = <8>
    # CHECK: lit.ref.store %[[IDX8]], %[[S_VAL]]
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[S_KEY]], %[[S_VAL]])

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
    # CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemOnly,
    # CHECK: %[[VALUE:.*]] = kgen.param.materialize: !MemOnly = <{}>
    # CHECK: store %[[VALUE]], %s
    var s = MemOnly()

    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl {{.*}}@OwnedKwargsDict<:!CollectionElement #[[MEM_ONLY]]>
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[Y_KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "y"}>
    # CHECK: %[[M:.*]] = lit.var.decl
    # CHECK: %[[VALUE:.*]] = kgen.param.materialize: !MemOnly = <{}>
    # CHECK: store %[[VALUE]], %[[M]]
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Y_KEY]], %[[M]])

    # CHECK: %[[S:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!MemOnly,
    # CHECK: %[[S_REF:.*]] = lit.ref.immut %s
    # CHECK: lit.call {{.*}}@MemOnly::@"__copyinit__{{.*}}(%[[S]], %[[S_REF]])
    # CHECK: %[[Z_KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "z"}>
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Z_KEY]], %[[S]])
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
    # CHECK-SAME: @OwnedKwargsDict<:!CollectionElement #[[MEM_ONLY]]>
    # CHECK: %[[RES:.*]] = lit.call {{.*}}@"takes_kw{{.*}}(%[[DICT_VAR]])
    # CHECK: lit.ref.store %[[RES]], %b
    var b = takes_kw(y=x, z=x)
