# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: Index):
    pass


# CHECK-LABEL: lit.fn @"function
fn function():
    # CHECK: materialize: !escaping{{.*}} = <{}>
    fn closure_with_loop(x: Index) escaping:
        if __mlir_attr.true:
            var t = x
            use(t)
