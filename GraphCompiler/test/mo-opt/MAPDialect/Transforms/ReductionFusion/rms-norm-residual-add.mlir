// RUN: mo-opt %s --mo-tmp-decompose-composites --canonicalize | mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels | FileCheck %s

// Two rms_norms with a residual added between them, the sum also an output
// (each rms_norm decomposed before the kernels are imported), fuse into two
// rowwise kernels. The first folds x and writes h = rms_norm(x) + r in a map
// phase; the second folds h and writes y. h is written once, by the first
// kernel.
// CHECK-LABEL: mo.graph @residual
// CHECK:       %[[H:.*]] = map.iter.rowwise<[8, 4096], axis = 1>
// CHECK:         map.iter.reduce
// CHECK:         map.iter.map
// CHECK:       %[[Y:.*]] = map.iter.rowwise<[8, 4096], axis = 1>
// CHECK-NEXT:  inputs  (%[[H]]: !mo.tensor<[8, 4096], bf16>
// CHECK:         map.iter.reduce
// CHECK:         map.iter.map
// CHECK-NOT:   map.iter.rowwise
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[Y]], %[[H]]
mo.graph @residual(%x: !mo.tensor<[8, 4096], bf16>, %r: !mo.tensor<[8, 4096], bf16>, %w1: !mo.tensor<[4096], bf16>, %w2: !mo.tensor<[4096], bf16>) -> (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[8, 4096], bf16>) {
  %n1 = mo.composite.reduce.rms_norm(%x, %w1) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16>
  %h = mo.add(%n1, %r) : !mo.tensor<[8, 4096], bf16>
  %y = mo.composite.reduce.rms_norm(%h, %w2) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16>
  mo.output %y, %h : !mo.tensor<[8, 4096], bf16>, !mo.tensor<[8, 4096], bf16>
}

// Without the sum as an output, the second rms_norm reads it, and so do only
// kernels that end up in it, through its per-row results: the sum is left to
// it as a producer and never written. The first kernel then runs at the start
// of the second's row.
// CHECK-LABEL: mo.graph @residualNotOutput
// CHECK:       %[[Y:.*]] = map.iter.rowwise<[8, 4096], axis = 1>
// CHECK-NOT:   map.iter.rowwise
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[Y]] :
mo.graph @residualNotOutput(%x: !mo.tensor<[8, 4096], bf16>, %r: !mo.tensor<[8, 4096], bf16>, %w1: !mo.tensor<[4096], bf16>, %w2: !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16> {
  %n1 = mo.composite.reduce.rms_norm(%x, %w1) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16>
  %h = mo.add(%n1, %r) : !mo.tensor<[8, 4096], bf16>
  %y = mo.composite.reduce.rms_norm(%h, %w2) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16>
  mo.output %y : !mo.tensor<[8, 4096], bf16>
}
