# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK:         lit.struct.field field0 : !Int
# CHECK:         lit.fn @"__init__{{.*}}*, %copy:


# CHECK-LABEL: lit.fn @"foo
def foo():
    var w = 5

    def bar() -> Int:
        var x = w + w
        return x
