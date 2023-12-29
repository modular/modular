# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

alias Int = __mlir_type.index

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

fn use(x: Int):
    pass


# CHECK-LABEL: lit.func @"function
fn function():
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%anonymous2A)
    fn closure_with_loop(x: Int) escaping:
        if __mlir_attr.`true`:
            let t = x
            use(t)
