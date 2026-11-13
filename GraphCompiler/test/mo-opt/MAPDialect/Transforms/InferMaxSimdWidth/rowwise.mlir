// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-infer-max-simd-width | FileCheck %s

// A rowwise kernel's width is limited by the ops in its folds and after them.
// A fold's index is the kernel's own, so the fold is not index computation;
// nothing but a concat limits the width.

// CHECK-LABEL: mo.graph @sum
// CHECK:       map.iter.rowwise.body<[4, 32], axis = 1>
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @sum(%x: !mo.tensor<[4, 32], f32>) -> !mo.tensor<[4, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[4, 32], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %r : !mo.tensor<[4, 1], f32>
}

// A fused producer in the fold and a per-row consumer after it.
// CHECK-LABEL: mo.graph @sum_sq_bias
// CHECK:       map.iter.rowwise.body<[4, 32], axis = 1>
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @sum_sq_bias(%x: !mo.tensor<[4, 32], f32>, %b: !mo.tensor<[4, 1], f32>) -> !mo.tensor<[4, 1], f32> {
  %sq = mo.mul(%x, %x) : !mo.tensor<[4, 32], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[4, 32], f32>) -> !mo.tensor<[4, 1], f32>
  %o = mo.add(%r, %b) : !mo.tensor<[4, 1], f32>
  mo.output %o : !mo.tensor<[4, 1], f32>
}

// A concat over the innermost dim, fused into the fold, limits the width to
// the GCD of its inputs' extents, as in a loop.
// CHECK-LABEL: mo.graph @sum_of_concat
// CHECK:       map.iter.rowwise.body<[4, 16], axis = 1> max_simd_width(2)
mo.graph @sum_of_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 1], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %r = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %r : !mo.tensor<[4, 1], f32>
}

// Over an outer concat axis: no limit.
// CHECK-LABEL: mo.graph @max_over_outer_concat_axis
// CHECK:       map.iter.rowwise.body<[8, 8], axis = 0>
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @max_over_outer_concat_axis(%a: !mo.tensor<[3, 8], f32>, %b: !mo.tensor<[5, 8], f32>) -> !mo.tensor<[1, 8], f32> {
  %c = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.reduce.max(%c) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[1, 8], f32>
  mo.output %r : !mo.tensor<[1, 8], f32>
}
