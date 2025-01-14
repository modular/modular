# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: Index):
    pass


# CHECK-LABEL: lit.fn @"direct
fn direct(output: Index):
    # CHECK: call {{.*}}_CI_{{.*}}__init__{{.*}}(%output, %anonymous2A)
    fn closure():
        @parameter
        fn body():
            if __mlir_attr.true:
                use(output)


# CHECK-LABEL: lit.fn @"deep_runtime_capture
fn deep_runtime_capture(
    m: Index,
) -> fn (n: Index) escaping -> fn (o: Index) escaping -> Index:
    # CHECK: lit.call {{.*}}_CI_{{.*}}__init__{{.*}}(%m, %anonymous2A)
    fn myclosure(n: Index) -> fn (o: Index) escaping -> Index:
        fn my_inner_closure(o: Index) -> Index:
            var x = __mlir_op.`index.add`(o, m)
            return __mlir_op.`index.add`(x, n)

        return my_inner_closure

    return myclosure
