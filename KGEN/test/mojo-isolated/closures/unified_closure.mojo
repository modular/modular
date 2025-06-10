# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# COM: Verify that the keyword parses but is not persisted in the IR.


# CHECK-NOT: my_closure
fn make_closure(x: Int):
    fn my_closure(y: Int) unified -> Int:
        return x + y
