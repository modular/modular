// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-infer-max-simd-width | FileCheck %s

// A rowwise kernel's width is limited by the ops in its folds and after them.
// A fold's index is the kernel's own, so the fold is not index computation;
// nothing in these kernels limits the width, so none gets a `max_simd_width`.

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
