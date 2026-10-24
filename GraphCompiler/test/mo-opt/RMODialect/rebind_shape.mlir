// RUN: mo-opt %s --split-input-file --canonicalize --verify-diagnostics | FileCheck %s

mo.graph @rebind_tensor_shape_canonicalize<A>(%arg0: !mo.tensor<[3, 4, A], f32>) -> !mo.tensor<[3, 4, A], f32> {
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<[3, 4, A], f32>) -> !mo.tensor<[3, 4, A], f32>
  mo.output %1 : !mo.tensor<[3, 4, A], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_canonicalize<A>([[VARG0:[^ ]*]]: !mo.tensor<[3, 4, A], f32>) -> !mo.tensor<[3, 4, A], f32> {
// CHECK-NEXT: mo.output [[VARG0]] : !mo.tensor<[3, 4, A], f32>

// -----

mo.graph @rebind_tensor_shape_not_ok(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<[], f32>) -> !mo.tensor<[], f64>
  mo.output %1 : !mo.tensor<[], f64>
}
