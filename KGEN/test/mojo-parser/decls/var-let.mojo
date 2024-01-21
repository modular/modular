# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index

alias `10` = __mlir_attr.`10 : index`
alias `42` = __mlir_attr.`42 : index`
alias `123` = __mlir_attr.`123 : index`


# COM: Stubs to allow testing without builtins
struct Error:
    pass


@register_passable("trivial")
struct Bool(AnyType):
    fn __mlir_i1__(self) -> __mlir_type.i1:
        pass


trait AnyType:
    pass


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


fn return_generic_memory_only[T: AnyType]() -> T:
    pass


fn fudge_int(x: Int) -> Int:
    return x


# CHECK-LABEL: lit.func @"let_decls()
fn let_decls():
    # CHECK: %x = lit.letreg.decl "x" = %index123 : index
    let x = `123`

    # CHECK: [[TMP:%.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%x)
    # CHECK: %z = lit.letreg.decl "z" = [[TMP]]
    let z = fudge_int(x)

    # These may be declared on the same line.
    let a = `42`
    let b = a
    # CHECK: %a = lit.letreg.decl "a" =
    # CHECK-NEXT: %b = lit.letreg.decl "b" =

    # COM: The parser cannot emit this into a `lit.letreg.decl` because the
    # COM: generic function call assumes memory-only conventions.
    # CHECK: lit.varlet.decl "c"
    let c = return_generic_memory_only[Bool]()


# CHECK-LABEL: lit.func @"var_decls()
fn var_decls():
    # CHECK: %y = lit.varlet.decl "y" var
    var y: Int

    # CHECK: %[[Y:.*]] = lit.ref.load %y
    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%[[Y]])
    # CHECK: lit.ref.store %[[F]], %y
    y = fudge_int(y)

    # CHECK: %z = lit.varlet.decl {{.*}} : !lit.ref<index,
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %y
    # CHECK-NEXT: lit.ref.store [[TMP]], %z
    var z = y
    z = `42`
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant = <42>
    # CHECK-NEXT: lit.ref.store [[TMP]], %z


# CHECK-LABEL: lit.func @"var_decls_implicit()
def var_decls_implicit() -> None:
    # Implicit declaration is mutable.
    # CHECK: %x = lit.varlet.decl "x" imp
    x = `123`

    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%index42)
    # CHECK: lit.ref.store %[[F]], %x
    x = fudge_int(`42`)


# CHECK-LABEL: lit.func @"test_var_let_scopes
fn test_var_let_scopes(cond: Bool):
    # CHECK: lit.letreg.decl "c"
    # CHECK: if
    let c = `10`
    if cond:
        # CHECK: lit.letreg.decl "c"
        let c = `42`
    # CHECK: else
    else:
        # CHECK: lit.letreg.decl "c"
        let c = `123`


# Issue #18157 and issue #18158, shadowing variables should be able to reference
# the shadowed variable on the RHS.
fn test_shadowing_reference_shadowed(cond: Bool):
    let num: Int = `10`
    if cond:
        let num = fudge_int(`42`)
