# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: Int):
    pass


# CHECK-LABEL: lit.fn @"direct
fn direct(output: Int):
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%output, %__call_result_tmp__)
    fn closure():
        @parameter
        fn body():
            if __mlir_attr.true:
                use(output)


# CHECK-LABEL: lit.fn @"deep_runtime_capture
fn deep_runtime_capture(
    m: Int,
) -> fn (n: Int) escaping -> fn (o: Int) escaping -> Int:
    # CHECK: lit.call {{.*}}_CI_{{.*}}__init__{{.*}}(%m, %__call_result_tmp__)
    fn myclosure(n: Int) -> fn (o: Int) escaping -> Int:
        fn my_inner_closure(o: Int) -> Int:
            var x = o + m
            return x + n

        return my_inner_closure

    return myclosure
