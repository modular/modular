# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s

# CHECK: lit.alias.decl *"z`0x0" = <0>
alias z = __mlir_attr.`0: index`


# CHECK-LABEL: lit.struct.decl @A<x, x_0>
struct A[x: Int, x_0: Int]:
    # CHECK: lit.alias.decl *"z`1x0" = <1>
    alias z = __mlir_attr.`1: index`
    # CHECK: lit.alias.decl *"y`1x1" = <11>
    alias y = __mlir_attr.`11: index`

    # CHECK-LABEL: lit.func @"foo
    # CHECK-SAME: <*"x`2x0", x_1>
    fn foo[x: Int, x_1: Int](self):
        # CHECK: lit.alias.decl *"z`2x1" = <2>
        alias z = __mlir_attr.`2: index`
        # CHECK: lit.alias.decl *"y`2x2" = <12>
        alias y = __mlir_attr.`12: index`
        # CHECK: lit.alias.decl *"yy`2x3" = <22>
        alias yy = __mlir_attr.`22: index`

        # CHECK-LABEL: lit.func *"bar
        # CHECK-SAME: <*"x`3x0", x_2>
        fn bar[x: Int, x_2: Int]():
            # CHECK: lit.alias.decl *"z`3x1" = <3>
            alias z = __mlir_attr.`3: index`


# COM: test names of implicit parameters
struct MyStruct[a: Int, b: Int]:
    pass


# CHECK-LABEL: lit.func @"test_implicit_parameters
# CHECK-SAME: <?, *"x`1x0", *"x`1x1", *"y`1x2", *"y`1x3">
fn test_implicit_parameters(x: MyStruct, y: MyStruct):
    pass
