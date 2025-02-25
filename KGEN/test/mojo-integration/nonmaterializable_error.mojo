# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo -debug-level full %s 2>&1 | FileCheck %s


fn main():
    var n: IntLiteral = 0
    for i in range(10):
        # CHECK: error: can't materialize IntLiteral in dynamic context
        # CHECK: error: failed to legalize operation 'kgen.param.constant' that was explicitly marked illegal
        n = n + 1
