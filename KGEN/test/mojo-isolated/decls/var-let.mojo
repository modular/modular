# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn return_generic_memory_only[T: AnyType]() -> T:
    pass


fn fudge_int(x: int) -> int:
    return x


# CHECK-LABEL: lit.func @"var_decls()
fn var_decls():
    # CHECK: %y = lit.var.decl "y" var
    var y: int

    # CHECK: %[[Y:.*]] = lit.ref.load %y
    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%[[Y]])
    # CHECK: lit.ref.store %[[F]], %y
    y = fudge_int(y)

    # CHECK: %z = lit.var.decl {{.*}} : !lit.ref<index,
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %y
    # CHECK-NEXT: lit.ref.store [[TMP]], %z
    var z = y
    z = `42`
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant = <42>
    # CHECK-NEXT: lit.ref.store [[TMP]], %z


# CHECK-LABEL: lit.func @"var_decls_implicit()
def var_decls_implicit() -> None:
    # Implicit declaration is mutable.
    # CHECK: %x = lit.var.decl "x" imp
    x = `123`

    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%index42)
    # CHECK: lit.ref.store %[[F]], %x
    x = fudge_int(`42`)


# CHECK-LABEL: lit.func @"test_var_let_scopes
fn test_var_let_scopes(cond: Bool):
    # CHECK: lit.var.decl "c"
    # CHECK: if
    var c = `10`
    if cond:
        # CHECK: lit.var.decl "c"
        var c = `42`
    # CHECK: else
    else:
        # CHECK: lit.var.decl "c"
        var c = `123`


# Issue #18157 and issue #18158, shadowing variables should be able to reference
# the shadowed variable on the RHS.
fn test_shadowing_reference_shadowed(cond: Bool):
    let num: int = `10`
    if cond:
        let num = fudge_int(`42`)
