# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK:         lit.struct.field field0 : !Int
# CHECK:         lit.fn @"__copyinit__


# CHECK-LABEL: lit.fn @"foo
fn foo():
    var w = 5

    fn bar() -> Int:
        var x = w + w
        return x
