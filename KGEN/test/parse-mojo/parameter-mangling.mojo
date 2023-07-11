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

# CHECK: kgen.param.declare _18x1_MY_NUMBER: {{.*}}@Int =
alias MY_NUMBER = 42


fn foo():
    # CHECK: kgen.param.declare _23x5_value: {{.*}}@Int =
    alias value = MY_NUMBER
