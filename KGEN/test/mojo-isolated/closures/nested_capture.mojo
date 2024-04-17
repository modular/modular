# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: int):
    pass


# CHECK-LABEL: lit.func @"direct
fn direct(output: int):
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%anonymous2A, %output)
    fn closure():
        @parameter
        fn body():
            if __mlir_attr.true:
                use(output)


# CHECK-LABEL: lit.func @"deep_runtime_capture
fn deep_runtime_capture(
    m: int,
) -> fn (n: int) escaping -> fn (o: int) escaping -> int:
    # CHECK: lit.call {{.*}}_CI_{{.*}}__init__{{.*}}(%anonymous2A, %m)
    fn myclosure(n: int) -> fn (o: int) escaping -> int:
        fn my_inner_closure(o: int) -> int:
            var x = __mlir_op.`index.add`(o, m)
            return __mlir_op.`index.add`(x, n)

        return my_inner_closure

    return myclosure
