# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: int):
    pass


# CHECK-LABEL: lit.func @"function
fn function():
    # CHECK: materialize: !escaping{{.*}} = <{}>
    fn closure_with_loop(x: int) escaping:
        if __mlir_attr.true:
            var t = x
            use(t)
