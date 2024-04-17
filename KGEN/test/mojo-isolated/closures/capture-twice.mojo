# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK:         lit.struct.field field0 : index
# CHECK:         lit.func @"__copyinit__


# CHECK-LABEL: lit.func @"foo
fn foo():
    var w = `5`

    fn bar() -> int:
        var x = __mlir_op.`index.add`(w, w)
        return x
