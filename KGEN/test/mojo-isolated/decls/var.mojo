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

# CHECK-LABEL: lit.func @"test_var_let_scopes
fn test_var_let_scopes(cond: Bool):
    # CHECK: lit.var.decl "c"
    # CHECK: hlcf.elif
    var c = `10`
    if cond:
        # CHECK: lit.var.decl "c"
        var c = `42`
    # CHECK: else
    else:
        # CHECK: lit.var.decl "c"
        var c = `123`


# CHECK-LABEL: lit.func @"test_var_lifetime_mangling
fn test_var_lifetime_mangling[x: int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<index, mut *"y`">
        var y = x
    # CHECK: } else {
    else:
        # CHECK: lit.var.decl "y" var : !lit.ref<index, mut *"y`1">
        var y = x


# CHECK-LABEL: lit.func @"test_nested_var_lifetime_mangling
fn test_nested_var_lifetime_mangling[x: int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<index, mut *"y`">
        var y = x

    # CHECK: lit.func *"nested()"
    fn nested():
        # CHECK: lit.var.decl "y" var : !lit.ref<index, mut *"y`2x">
        var y = x


# Issue #18157 and issue #18158, shadowing variables should be able to reference
# the shadowed variable on the RHS.
fn test_shadowing_reference_shadowed(cond: Bool):
    var num: int = `10`
    if cond:
        var num = fudge_int(`42`)

# ===----------------------------------------------------------------------=== #
# Implicitly declared variables.
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.func @"var_decls_implicit()
def var_decls_implicit() -> None:
    # Implicit declaration is mutable.
    # CHECK: %x = lit.var.decl "x" imp
    x = `123`

    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%index42)
    # CHECK: lit.ref.store %[[F]], %x
    x = fudge_int(`42`)


fn use_int(x: Int): pass

# Check implicit values are declared at top level where they belong.
# https://github.com/modularml/modular/issues/34368

# CHECK-LABEL: lit.func @"walrus_control_flow
def walrus_control_flow(a: Int):
   # CHECK: %b = lit.var.decl
   # CHECK: %curr = lit.var.decl "curr"
   curr = a

   # CHECK: lit.loop cond {
   # CHECK-NEXT: lit.ref.load %curr
   while b := curr + 1:
   # CHECK: } body {
   # CHECK-NEXT: lit.ref.load %b
     use_int(b)
     curr = b

# Check that we only get one implicit declaration and all three scopes use it.
# CHECK-LABEL: lit.func @"reuse_implicit
def reuse_implicit(a: Int, cond: __mlir_type.i1):
  # CHECK: %implicit = lit.var.decl

  # CHECK: hlcf.elif
  if cond:
      # CHECK: lit.ref.store %a, %implicit :
      implicit = a
      # CHECK: lit.ref.load %implicit :
      use_int(implicit)

  # CHECK: hlcf.elif
  if cond:
      # CHECK: lit.ref.store %a, %implicit :
      implicit = a
      # CHECK: lit.ref.load %implicit :
      use_int(implicit)

  # CHECK: lit.ref.store %a, %implicit :
  implicit = a
  # CHECK: lit.ref.load %implicit :
  use_int(implicit)
