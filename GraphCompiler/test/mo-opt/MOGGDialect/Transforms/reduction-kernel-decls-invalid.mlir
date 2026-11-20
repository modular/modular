// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%reduction_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s

// Each monoid in the reduction_invalid_stubs package breaks one rule from
// `ReduceOp`'s docstring. Verification reports all of them in one run.

// CHECK-DAG: error: reduction 'invalid_no_outputs' has no output fields (SIMD fields whose width is `W`)
// CHECK-DAG: error: reduction 'invalid_width_literal': the `width` alias must name one of the struct's parameters
// CHECK-DAG: error: reduction 'invalid_width_expr': the `width` alias must name one of the struct's parameters
// CHECK-DAG: error: reduction 'invalid_uninferable_param': parameter 'K' is neither the `width` parameter, an output's dtype, nor defaulted
// CHECK-DAG: error: reduction 'invalid_entry_point' must not define a kernel entry point

module {}
