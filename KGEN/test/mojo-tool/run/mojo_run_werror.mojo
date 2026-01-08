# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror converts warnings to errors for `mojo run`.

# RUN: not mojo run -Werror %s 2>&1 | FileCheck %s

# CHECK: error: assignment to 'foo' was never used
# CHECK-NOT: warning: assignment to 'foo' was never used


def main():
    var foo = 1
