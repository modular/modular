# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror converts warnings to errors for `mojo build`.

# RUN: not mojo build -Werror %s -o %t 2>&1 | FileCheck %s

# CHECK: error: assignment to 'foo' was never used
# CHECK-NOT: warning: assignment to 'foo' was never used


def main():
    var foo = 1
