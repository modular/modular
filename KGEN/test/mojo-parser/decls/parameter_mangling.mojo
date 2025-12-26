# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.alias.decl *"z`0x" = <0>
comptime z = __mlir_attr.`0: index`


# CHECK-LABEL: lit.struct.decl @A<x: !Int, x_0: !Int>
struct A[x: Int, x_0: Int]:
    # CHECK: lit.alias.decl *"z`" = <1>
    comptime z = __mlir_attr.`1: index`
    # CHECK: lit.alias.decl *"y`1" = <11>
    comptime y = __mlir_attr.`11: index`

    # CHECK-LABEL: lit.fn @"foo
    # CHECK-SAME: <_x: !Int, x_1: !Int>[imm *"self`2x"]
    fn foo[_x: Int, x_1: Int](self):
        # CHECK: lit.alias.decl *"z`2x1" = <2>
        comptime z = __mlir_attr.`2: index`
        # CHECK: lit.alias.decl *"y`2x2" = <12>
        comptime y = __mlir_attr.`12: index`
        # CHECK: lit.alias.decl *"yy`2x3" = <22>
        comptime yy = __mlir_attr.`22: index`

        # CHECK-LABEL: lit.fn *"bar{{.*}}<*"x`3x": !Int, x_2: !Int>
        fn bar[x: Int, x_2: Int]():
            # CHECK: lit.alias.decl *"z`3x1" = <3>
            comptime z = __mlir_attr.`3: index`


# COM: test names of implicit parameters
struct MyStruct[a: Int, b: Int]:
    pass


# CHECK-LABEL: lit.fn @"test_implicit_parameters
# CHECK-SAME: <?, *"x.a`": !Int, *"x.b`1": !Int, *"y.a`3": !Int, *"y.b`4": !Int>[imm *"x`2", imm *"y`5"]
fn test_implicit_parameters(x: MyStruct, y: MyStruct):
    pass


# CHECK-LABEL: lit.fn @"test_nested_alias_mangling_1
fn test_nested_alias_mangling_1[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        comptime y = x
        _ = y
    # CHECK: } else {
    else:
        # CHECK: lit.alias.decl *"y`1"
        comptime y = x
        _ = y


# CHECK-LABEL: lit.fn @"test_nested_alias_mangling_2
fn test_nested_alias_mangling_2[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        comptime y = x
        _ = y

    # CHECK: lit.fn *"nested()"
    fn nested():
        # CHECK: lit.alias.decl *"y`2x"
        comptime y = x
        _ = y
