# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK-NEXT: lit.struct.field field0 : index
# CHECK: lit.func @"__copyinit__


# CHECK-LABEL: lit.func @"foo
fn foo():
    let w = `5`

    fn bar() escaping -> Int:
        let x = __mlir_op.`index.add`(w, w)
        return x
