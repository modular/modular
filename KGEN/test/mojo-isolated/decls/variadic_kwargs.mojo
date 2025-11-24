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

# CHECK-LABEL: lit.fn @"variadic_kwargs
# CHECK-SAME: "[mut [[LT:.*]]](
# CHECK-SAME: %a: !Int, %b: !Int, %args: !kgen.variadic<!Int> pos_vararg, *, %c: !Int, %d: !Int,
# CHECK-SAME: %kwargs: !lit.ref<!lit.struct<#OwnedKwargsDict <:!ImplicitlyCopyable_Movable !Int>>, mut [[LT]]> owned_in_mem|kw_vararg)
fn variadic_kwargs(
    a: Int, b: Int, *args: Int, c: Int, d: Int, **kwargs: Int
):
    pass


# CHECK-LABEL: lit.fn @"variadic_kwargs_def_with_type
def variadic_kwargs_def_with_type(**kwargs: Int):
    pass


# CHECK-SAME: "[mut [[LT:.*]], {{.*}}](*, %kwargs: !lit.ref<!lit.struct<#OwnedKwargsDict <:!ImplicitlyCopyable_Movable !Int>>, mut [[LT]]> owned_in_mem|kw_vararg,
fn takes_int_variadic_kwargs(**kwargs: Int):
    pass


fn takes_int_variadic_kwargs_multiline(
    **kwargs: Int,
):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_kwargs
fn test_variadic_kwargs():
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: !lit.struct<#OwnedKwargsDict <:!ImplicitlyCopyable_Movable !Int>>
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[X_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "x">
    # CHECK: %[[X_VAL:.*]] = lit.var.decl {{.*}}!Int,
    # CHECK: %[[IDX9:.*]] = kgen.param.constant: !Int = <{9}>
    # CHECK: lit.ref.store %[[IDX9]], %[[X_VAL]]
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[X_KEY]], %[[X_VAL]])

    # CHECK: %[[S_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "stuff">
    # CHECK: %[[S_VAL:.*]] = lit.var.decl {{.*}}!Int,
    # CHECK: %[[IDX8:.*]] = kgen.param.constant: !Int = <{8}>
    # CHECK: lit.ref.store %[[IDX8]], %[[S_VAL]]
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[S_KEY]], %[[S_VAL]])

    # CHECK lit.call {{.*}}@"takes_int_variadic_kwargs{{.*}}(%[[DICT_VAR]])
    takes_int_variadic_kwargs(x=9, stuff=8)


trait SomeTrait(ImplicitlyCopyable, Movable):
    pass


fn infers_param_from_kwargs[T: SomeTrait](**kwargs: T):
    pass


@fieldwise_init
struct MemOnly(SomeTrait):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_kwargs_param_inference
fn test_variadic_kwargs_param_inference():
    # CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemOnly,
    # CHECK: lit.call {{.*}}MemOnly::@"__init__{{.*}}(%s)
    var s = MemOnly()

    # CHECK: %[[M:.*]] = lit.var.decl
    # CHECK: lit.call {{.*}}MemOnly::@"__init__{{.*}}(%[[M]])

    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl {{.*}}#OwnedKwargsDict <:!ImplicitlyCopyable_Movable !MemOnly>
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"__init__{{.*}}(%[[DICT_VAR]])
    # CHECK: %[[Y_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "y">

    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Y_KEY]], %[[M]])

    # CHECK: %[[S_REF:.*]] = lit.ref.immut %s
    # CHECK: %[[S:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!MemOnly,
    # CHECK: lit.call {{.*}}@MemOnly::@"__copyinit__{{.*}}(%[[S_REF]], %[[S]])
    # CHECK: %[[Z_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "z">
    # CHECK: lit.call {{.*}}@OwnedKwargsDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Z_KEY]], %[[S]])
    infers_param_from_kwargs(y=MemOnly(), z=s)


# COM: test that the inferred type of variables is correct when the initializer
# COM: expression has variadic keyword arguments.
# COM: Issue https://github.com/modularml/modular/issues/35215
fn takes_kw(**kwargs: MemOnly) -> Int:
    return 0


# CHECK-LABEL: lit.fn @"test_takes_kw_in_assignment
fn test_takes_kw_in_assignment(x: MemOnly):
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl{{.*}}#OwnedKwargsDict <:!ImplicitlyCopyable_Movable !MemOnly>
    # CHECK: %[[RES:.*]] = lit.call {{.*}}@"takes_kw{{.*}}(%[[DICT_VAR]])
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!Int,
    # CHECK: lit.ref.store %[[RES]], %b
    var b = takes_kw(y=x, z=x)
