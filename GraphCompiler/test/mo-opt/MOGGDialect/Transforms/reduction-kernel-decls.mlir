// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mogg-verify-kernels --mogg-dump-kernels 2>&1 | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" | mo-opt --mogg-verify-kernels --mogg-dump-kernels 2>&1 | FileCheck %s

// A registered `ReduceOp` monoid is a reduction. Its width parameter is the
// one the trait's `width` alias names, and its outputs are its `SIMD` fields
// of that width.

// CHECK-DAG: reduction mo.reduce.add: width = W, outputs = [acc]
// CHECK-DAG: reduction mo.reduce.max: width = W, outputs = [acc]
// CHECK-DAG: reduction mo.reduce.min: width = W, outputs = [acc]
// CHECK-DAG: reduction mo.reduce.mul.tmp: width = W, outputs = [acc]
// CHECK-DAG: reduction mo.reduce.reduce_min_and_max.tmp: width = W, outputs = [min_acc, max_acc]
// CHECK-DAG: reduction mo.reduce.arg_max.tmp: width = W, outputs = [acc_values, acc_indices]
// CHECK-DAG: reduction mo.reduce.arg_min.tmp: width = W, outputs = [acc_values, acc_indices]
// CHECK-DAG: reduction mo.reduce.mean_var.tmp: width = W, outputs = [count, mean, M2]
// CHECK-DAG: reduction mo.reduce.logsumexp.tmp: width = W, outputs = [m, l]

// `best` (`Scalar`), `pair` (width 2) and `count` (`Int`) are internal.
// CHECK-DAG: reduction test_monoid_mixed_fields: width = W, outputs = [acc, index]

// Nothing depends on parameter names.
// CHECK-DAG: reduction test_monoid_renamed_params: width = V, outputs = [total]

// An extra parameter with a default is allowed.
// CHECK-DAG: reduction test_monoid_defaulted_param: width = W, outputs = [acc]

module {}
