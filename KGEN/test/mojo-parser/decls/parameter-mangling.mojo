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

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# CHECK: lit.alias.decl *"z`0x1" = <0>
alias z = __mlir_attr.`0: index`


# CHECK-LABEL: lit.struct.decl @A
struct A[x: Int, x_0: Int]:
    # CHECK: lit.alias.decl *"z`1x0" = <1>
    alias z = __mlir_attr.`1: index`
    # CHECK: lit.alias.decl *"y`1x1" = <11>
    alias y = __mlir_attr.`11: index`

    fn foo[x: Int, x_1: Int](self):
        # CHECK: lit.alias.decl *"z`2x0" = <2>
        alias z = __mlir_attr.`2: index`
        # CHECK: lit.alias.decl *"y`2x1" = <12>
        alias y = __mlir_attr.`12: index`
        # CHECK: lit.alias.decl *"yy`2x2" = <22>
        alias yy = __mlir_attr.`22: index`

        fn bar[x: Int, x_2: Int]():
            # CHECK: lit.alias.decl *"z`3x0" = <3>
            alias z = __mlir_attr.`3: index`
