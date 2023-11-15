# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

alias Int = __mlir_type.index


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
