# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s

alias Int = __mlir_type.index


fn use(x: Int):
    pass


# CHECK-LABEL: lit.func @"function
fn function():
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%anonymous2A)
    fn closure_with_loop(x: Int) escaping:
        if __mlir_attr.`true`:
            let t = x
            use(t)
