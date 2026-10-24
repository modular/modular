// RUN: mo-opt --canonicalize %s | FileCheck %s

// Test canonicalization pattern that folds scatter-like operations with empty indices tensors.
// These operations become no-ops when indices is empty, so they should be replaced with their input.

// CHECK-LABEL: @fold_scatter_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[4, 5], f32>, %[[UPDATES:.+]]: !mo.tensor<[0, 5], f32>, %[[INDICES:.+]]: !mo.tensor<[0, 5], si32>)
mo.graph @fold_scatter_empty_indices(%input: !mo.tensor<[4, 5], f32>, %updates: !mo.tensor<[0, 5], f32>, %indices: !mo.tensor<[0, 5], si32>) -> !mo.tensor<[4, 5], f32> {
  // CHECK-NOT: mo.scatter
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter(%input, %updates, %indices) [axis = 0] : (!mo.tensor<[4, 5], f32>, !mo.tensor<[0, 5], f32>, !mo.tensor<[0, 5], si32>) -> !mo.tensor<[4, 5], f32>
  mo.output %result : !mo.tensor<[4, 5], f32>
}

// CHECK-LABEL: @fold_scatter_add_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[3, 4], si32>, %[[UPDATES:.+]]: !mo.tensor<[0, 4], si32>, %[[INDICES:.+]]: !mo.tensor<[0, 4], si64>, %[[AXIS:.+]]: !mo.tensor<[], si32>)
mo.graph @fold_scatter_add_empty_indices(%input: !mo.tensor<[3, 4], si32>, %updates: !mo.tensor<[0, 4], si32>, %indices: !mo.tensor<[0, 4], si64>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<[3, 4], si32> {
  // CHECK-NOT: mo.scatter.add
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 4], si32>, !mo.tensor<[0, 4], si32>, !mo.tensor<[0, 4], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[3, 4], si32>
  mo.output %result : !mo.tensor<[3, 4], si32>
}

// CHECK-LABEL: @fold_scatter_max_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[2, 3, 4], f16>, %[[UPDATES:.+]]: !mo.tensor<[2, 0, 4], f16>, %[[INDICES:.+]]: !mo.tensor<[2, 0, 4], si32>, %[[AXIS:.+]]: !mo.tensor<[], si32>)
mo.graph @fold_scatter_max_empty_indices(%input: !mo.tensor<[2, 3, 4], f16>, %updates: !mo.tensor<[2, 0, 4], f16>, %indices: !mo.tensor<[2, 0, 4], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<[2, 3, 4], f16> {
  // CHECK-NOT: mo.scatter.max
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[2, 3, 4], f16>, !mo.tensor<[2, 0, 4], f16>, !mo.tensor<[2, 0, 4], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 3, 4], f16>
  mo.output %result : !mo.tensor<[2, 3, 4], f16>
}

// CHECK-LABEL: @fold_scatter_min_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[5], ui8>, %[[UPDATES:.+]]: !mo.tensor<[0], ui8>, %[[INDICES:.+]]: !mo.tensor<[0], si32>, %[[AXIS:.+]]: !mo.tensor<[], si32>)
mo.graph @fold_scatter_min_empty_indices(%input: !mo.tensor<[5], ui8>, %updates: !mo.tensor<[0], ui8>, %indices: !mo.tensor<[0], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<[5], ui8> {
  // CHECK-NOT: mo.scatter.min
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[5], ui8>, !mo.tensor<[0], ui8>, !mo.tensor<[0], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5], ui8>
  mo.output %result : !mo.tensor<[5], ui8>
}

// CHECK-LABEL: @fold_scatter_mul_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[2, 2], f64>, %[[UPDATES:.+]]: !mo.tensor<[0, 2], f64>, %[[INDICES:.+]]: !mo.tensor<[0, 2], si64>, %[[AXIS:.+]]: !mo.tensor<[], si32>)
mo.graph @fold_scatter_mul_empty_indices(%input: !mo.tensor<[2, 2], f64>, %updates: !mo.tensor<[0, 2], f64>, %indices: !mo.tensor<[0, 2], si64>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<[2, 2], f64> {
  // CHECK-NOT: mo.scatter.mul
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[2, 2], f64>, !mo.tensor<[0, 2], f64>, !mo.tensor<[0, 2], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 2], f64>
  mo.output %result : !mo.tensor<[2, 2], f64>
}

// CHECK-LABEL: @no_fold_scatter_non_empty_indices
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[4, 5], f32>, %[[UPDATES:.+]]: !mo.tensor<[2, 5], f32>, %[[INDICES:.+]]: !mo.tensor<[2, 5], si32>)
// Test that the pattern doesn't fold when indices are non-empty
mo.graph @no_fold_scatter_non_empty_indices(%input: !mo.tensor<[4, 5], f32>, %updates: !mo.tensor<[2, 5], f32>, %indices: !mo.tensor<[2, 5], si32>) -> !mo.tensor<[4, 5], f32> {
  // CHECK: %[[RESULT:.+]] = mo.scatter
  %result = mo.scatter(%input, %updates, %indices) [axis = 0] : (!mo.tensor<[4, 5], f32>, !mo.tensor<[2, 5], f32>, !mo.tensor<[2, 5], si32>) -> !mo.tensor<[4, 5], f32>
  // CHECK: mo.output %[[RESULT]]
  mo.output %result : !mo.tensor<[4, 5], f32>
}

// CHECK-LABEL: @fold_dynamic_shape_scatter_empty_indices<D0, D1>
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[D0, D1], f32>, %[[UPDATES:.+]]: !mo.tensor<[0, D1], f32>, %[[INDICES:.+]]: !mo.tensor<[0, D1], si32>)
// Test with symbolic dimensions
mo.graph @fold_dynamic_shape_scatter_empty_indices<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[0, D1], f32>, %indices: !mo.tensor<[0, D1], si32>) -> !mo.tensor<[D0, D1], f32> {
  // CHECK-NOT: mo.scatter
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter(%input, %updates, %indices) [axis = 0] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[0, D1], f32>, !mo.tensor<[0, D1], si32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: @fold_scatter_add_empty_indices_axis_1
// CHECK: (%[[INPUT:.+]]: !mo.tensor<[3, 4, 5], si16>, %[[UPDATES:.+]]: !mo.tensor<[3, 0, 5], si16>, %[[INDICES:.+]]: !mo.tensor<[3, 0, 5], si32>, %[[AXIS:.+]]: !mo.tensor<[], si32>)
// Test with different axis value
mo.graph @fold_scatter_add_empty_indices_axis_1(%input: !mo.tensor<[3, 4, 5], si16>, %updates: !mo.tensor<[3, 0, 5], si16>, %indices: !mo.tensor<[3, 0, 5], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<[3, 4, 5], si16> {
  // CHECK-NOT: mo.scatter.add
  // CHECK: mo.output %[[INPUT]]
  %result = mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 4, 5], si16>, !mo.tensor<[3, 0, 5], si16>, !mo.tensor<[3, 0, 5], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[3, 4, 5], si16>
  mo.output %result : !mo.tensor<[3, 4, 5], si16>
}
