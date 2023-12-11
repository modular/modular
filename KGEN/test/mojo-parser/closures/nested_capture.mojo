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


# CHECK-LABEL: lit.func @"direct
fn direct(output: Int):
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%0, %arg)
    fn closure() escaping:
        @parameter
        fn body():
            if __mlir_attr.`true`:
                use(output)


# CHECK-LABEL: lit.func @"deep_runtime_capture
fn deep_runtime_capture(
    m: __mlir_type.index,
) -> fn (n: __mlir_type.index) escaping -> fn (o: __mlir_type.index) escaping -> __mlir_type.index:
    # CHECK: lit.call {{.*}}_CI_{{.*}}__init__{{.*}}(%0, %m)
    fn myclosure(n: __mlir_type.index) escaping -> fn (o: __mlir_type.index) escaping -> __mlir_type.index:
       fn my_inner_closure(o: __mlir_type.index) escaping -> __mlir_type.index:
         let x = __mlir_op.`index.add`(o, m)
         return __mlir_op.`index.add`(x, n)
       return my_inner_closure

    return myclosure
