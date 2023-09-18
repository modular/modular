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

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# COM: This file

# CHECK: lit.alias.decl _18x1_MY_NUMBER: !IntLiteral =
alias MY_NUMBER = 42


fn foo():
    # CHECK: lit.alias.decl _23x5_value: !IntLiteral =
    alias value = MY_NUMBER
