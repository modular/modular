# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file contains tests for parameter name mangling in the parser. These
# tests are in a standalone file, because they are fairly brittle due to the
# name mangling using the declarations location.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.alias.decl _16x1_MY_NUMBER
alias MY_NUMBER = __mlir_attr.`42 : index`


fn foo():
    # CHECK: lit.alias.decl _21x5_value
    alias value = MY_NUMBER
