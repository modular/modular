# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


def use(x: Int):
    pass


# CHECK-LABEL: lit.fn @"direct
def direct(output: Int):
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%output, %__call_result_tmp__)
    def closure():
        @parameter
        def body():
            if __mlir_attr.true:
                use(output)


# CHECK-LABEL: lit.fn @"deep_runtime_capture
def deep_runtime_capture(
    m: Int,
) -> def (n: Int) escaping -> def (o: Int) escaping -> Int:
    # CHECK: lit.call {{.*}}_CI_{{.*}}__init__{{.*}}(%m, %__call_result_tmp__)
    def myclosure(n: Int) -> def (o: Int) escaping -> Int:
        def my_inner_closure(o: Int) -> Int:
            var x = o + m
            return x + n

        return my_inner_closure

    return myclosure
