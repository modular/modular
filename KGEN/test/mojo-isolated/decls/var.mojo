# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct MemExample:
    pass


fn return_generic_memory_only[T: AnyType]() -> T:
    pass


fn fudge_int(x: Int) -> Int:
    return x


# CHECK-LABEL: lit.fn @"var_decls()
fn var_decls():
    # CHECK: %y = lit.var.decl "y" var
    var y: Int

    # CHECK: %[[Y:.*]] = lit.ref.load %y
    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%[[Y]])
    # CHECK: lit.ref.store %[[F]], %y
    y = fudge_int(y)

    # CHECK: %z = lit.var.decl {{.*}} : !lit.ref<!Int,
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %y
    # CHECK-NEXT: lit.ref.store [[TMP]], %z
    var z = y
    z = 42
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK-NEXT: lit.ref.store [[TMP]], %z


# CHECK-LABEL: lit.fn @"test_var_let_scopes
fn test_var_let_scopes(cond: Bool):
    # CHECK: lit.var.decl "c"
    # CHECK: hlcf.elif
    var c = 10
    if cond:
        # CHECK: lit.var.decl "c"
        var c = 42
    # CHECK: else
    else:
        # CHECK: lit.var.decl "c"
        var c = 123


# CHECK-LABEL: lit.fn @"test_var_origin_mangling
fn test_var_origin_mangling[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`">
        var y = x
    # CHECK: } else {
    else:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`1">
        var y = x


# CHECK-LABEL: lit.fn @"test_nested_var_origin_mangling
fn test_nested_var_origin_mangling[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`">
        var y = x

    # CHECK: lit.fn *"nested()"
    fn nested():
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`2x">
        var y = x


# Issue #18157 and issue #18158, shadowing variables should be able to reference
# the shadowed variable on the RHS.
fn test_shadowing_reference_shadowed(cond: Bool):
    var num: Int = 10
    if cond:
        var num = fudge_int(42)


# ===----------------------------------------------------------------------=== #
# Implicitly declared variables.
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"var_decls_implicit()
def var_decls_implicit() -> None:
    # Implicit declaration is mutable.
    # CHECK: %x = lit.var.decl "x" imp
    x = 123

    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: [[F:%.*]] = lit.call {{.*}}::@"fudge_int{{.*}}([[TMP]])
    # CHECK: lit.ref.store [[F]], %x
    x = fudge_int(42)


fn use_int(x: Int):
    pass


# Check implicit values are declared at top level where they belong.
# https://github.com/modularml/modular/issues/34368


# CHECK-LABEL: lit.fn @"walrus_control_flow
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
# CHECK-LABEL: lit.fn @"reuse_implicit
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


# CHECK-LABEL: lit.fn @"addrSpaces
fn addrSpaces[lt1: MutOrigin, lt2: ImmutOrigin, as1: AddressSpace]():
    # CHECK: lit.var.decl "ref1" {{.*}}!lit.ref<!MemExample, mut {{.*}}lt1{{.*}}, #lit.struct.extract<:!Int #lit.struct.extract<:!AddressSpace as1, "_value">, "_mlir_value">>
    var ref1: Pointer[MemExample, lt1, as1]._mlir_type

    # CHECK: lit.alias.decl [[AS2:.*]]: !AddressSpace = {{.*}} {42}
    comptime as2: AddressSpace = AddressSpace(42)

    # CHECK: lit.var.decl "ref2" {{.*}}!lit.ref<!MemExample, imm #lit.struct.extract<:!lit.struct<#Origin <:!Bool {:i1 0}>> lt2, "_mlir_origin">, 42>
    var ref2: __mlir_type[
        `!lit.ref<`, MemExample, `, `, lt2._mlir_origin, `, `, +as2._value._mlir_value, `>`
    ]
