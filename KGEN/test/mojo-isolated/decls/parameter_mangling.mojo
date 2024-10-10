# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.alias.decl *"z`0x" = <0>
alias z = __mlir_attr.`0: index`


# CHECK-LABEL: lit.struct.decl @A<x, x_0>
struct A[x: int, x_0: int]:
    # CHECK: lit.alias.decl *"z`" = <1>
    alias z = __mlir_attr.`1: index`
    # CHECK: lit.alias.decl *"y`1" = <11>
    alias y = __mlir_attr.`11: index`

    # CHECK-LABEL: lit.func @"foo
    # CHECK-SAME: <*"x`2x", x_1>[imm *"self`2x1"]
    fn foo[x: int, x_1: int](self):
        # CHECK: lit.alias.decl *"z`2x2" = <2>
        alias z = __mlir_attr.`2: index`
        # CHECK: lit.alias.decl *"y`2x3" = <12>
        alias y = __mlir_attr.`12: index`
        # CHECK: lit.alias.decl *"yy`2x4" = <22>
        alias yy = __mlir_attr.`22: index`

        # CHECK-LABEL: lit.func *"bar
        # CHECK-SAME: <*"x`3x", x_2>
        fn bar[x: int, x_2: int]():
            # CHECK: lit.alias.decl *"z`3x1" = <3>
            alias z = __mlir_attr.`3: index`


# COM: test names of implicit parameters
struct MyStruct[a: int, b: int]:
    pass


# CHECK-LABEL: lit.func @"test_implicit_parameters
# CHECK-SAME: <?, *"a`", *"b`1", *"a`3", *"b`4">[imm *"x`2", imm *"y`5"]
fn test_implicit_parameters(x: MyStruct, y: MyStruct):
    pass


# CHECK-LABEL: lit.func @"test_nested_alias_mangling_1
fn test_nested_alias_mangling_1[x: int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        alias y = x
        _ = y
    # CHECK: } else {
    else:
        # CHECK: lit.alias.decl *"y`1"
        alias y = x
        _ = y


# CHECK-LABEL: lit.func @"test_nested_alias_mangling_2
fn test_nested_alias_mangling_2[x: int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        alias y = x
        _ = y

    # CHECK: lit.func *"nested()"
    fn nested():
        # CHECK: lit.alias.decl *"y`2x"
        alias y = x
        _ = y
