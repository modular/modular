# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s


fn make_closure(x: Int):
    # CHECK: error: expected a capture convention list
    fn my_closure(y: Int) unified -> Int:
        return x + y
