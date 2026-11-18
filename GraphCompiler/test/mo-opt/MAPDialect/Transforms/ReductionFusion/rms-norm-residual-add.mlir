// RUN: mo-opt %s --mo-tmp-decompose-composites --canonicalize | mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels | FileCheck %s

// Two rms_norms with a residual added between them, the sum also an output
// (each rms_norm decomposed before the kernels are imported), fuse into one
// rowwise kernel: fold x, cache h = rms_norm(x) + r and store it, fold h from
// the cache, and write y from it.
// CHECK-LABEL: mo.graph @residual
// CHECK:       %[[R:.*]]:2 = map.iter.rowwise<[8, 4096], axis = 1>
// CHECK:         outputs (%[[Y:.*]]: !mo.tensor<[8, 4096], bf16>, %[[H:.*]]: !mo.tensor<[8, 4096], bf16>)
// CHECK:         map.iter.reduce
// CHECK:         %[[C:.*]] = map.iter.cache index (%[[K:.*]]: !map.index<2>) -> bf16 {
// CHECK:           %[[HV:.*]] = map.call.elementwise @builtin_kernels.elementwise.Add(%[[K]],
// CHECK-NEXT:      map.store %[[H]][%[[K]]] = %[[HV]]
// CHECK-NEXT:      map.yield %[[HV]] : bf16
// CHECK:         map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[I:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      map.iter.cache.load %[[C]][%[[I]]]
// CHECK:         map.iter.map index (%[[J:.*]]: !map.index<2>) {
// CHECK-NEXT:      map.iter.cache.load %[[C]][%[[J]]]
// CHECK:           map.store %[[Y]][%[[J]]]
// CHECK-NOT:   map.iter.rowwise
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[R]]#0, %[[R]]#1
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

// As in a transformer layer, h also feeds the next residual add, after a
// hand-written kernel that reads y. h is still one kernel's cached and stored
// output, which the add reads.
// CHECK-LABEL: mo.graph @residualOfNextLayer
// CHECK:       %[[R2:.*]]:2 = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:         map.iter.cache
// CHECK:           map.store
// CHECK:       %[[A2:.*]] = map.iter.opaque
// CHECK:       map.iter.elementwise
// CHECK-NEXT:  inputs  (%[[R2]]#1: !mo.tensor<[8, 128], f32>, %[[A2]]: !mo.tensor<[8, 128], f32>)
// CHECK-NOT:   map.iter.rowwise
mo.graph @residualOfNextLayer(%x: !mo.tensor<[8, 128], f32>, %r: !mo.tensor<[8, 128], f32>, %w1: !mo.tensor<[128], f32>, %w2: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32> {
  %n1 = mo.composite.reduce.rms_norm(%x, %w1) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], f32>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32>
  %h = mo.add(%n1, %r) : !mo.tensor<[8, 128], f32>
  %y = mo.composite.reduce.rms_norm(%h, %w2) {epsilon = #M.dense_array<1.0e-05> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 128], f32>, !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32>
  %a = mo.custom [symbol = "map_lower_partial_fused_inputs", device = #M.device_ref<"cpu", 0>](%r, %y) : (!mo.tensor<[8, 128], f32>, !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32>
  %z = mo.add(%h, %a) : !mo.tensor<[8, 128], f32>
  mo.output %z : !mo.tensor<[8, 128], f32>
}
