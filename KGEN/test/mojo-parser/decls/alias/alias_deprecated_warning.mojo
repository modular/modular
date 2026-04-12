# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Test that the 'alias' keyword issues a deprecation warning

# CHECK: warning: 'alias' is deprecated; use 'comptime'
alias MY_CONSTANT = 42


struct MyStruct:
    # CHECK: warning: 'alias' is deprecated; use 'comptime'
    alias SIZE = Int


# Test that 'comptime' does NOT issue a warning about being deprecated
# CHECK-NOT: comptime NEW_CONSTANT
comptime NEW_CONSTANT = 99
