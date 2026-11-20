// RUN: mo-opt %s --split-input-file --mo-support-conversions-test --mo-legalize-rmo-operators --mo-constant-fold --mo-shape-inference --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_prelu() -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<-4.000000e-01, -2.000000e-01, 0.000000e+00, 1.000000e+00, 2.000000e+00> : !M.array<5xf32>> : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[5], f32>
mo.graph @support_convert_prelu() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-2.0, -1.0, 0.0, 1.0, 2.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<0.2> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[5], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_prelu_mixed_types() -> !mo.tensor<[3], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<2, 1, 3> : !M.array<3xsi32>> : !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[3], si32>
mo.graph @support_convert_prelu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant <#M.dense_array<2, 1, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %1 = mo.constant <#M.dense_array<0.3> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[3], si32>, !mo.tensor<[1], f32>) -> !mo.tensor<[3], si32>
  mo.output %2 : !mo.tensor<[3], si32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu() -> !mo.tensor<[3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<0.000000e+00, 0.000000e+00, 5.000000e+00> : !M.array<3xf32>> : !mo.tensor<[3], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[3], f32>
mo.graph @support_convert_relu() -> (!mo.tensor<[3], f32>) {
  %0 = mo.constant <#M.dense_array<-3.0, 0.0, 5.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %1 = "relu"(%0) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], f32>
  mo.output %1 : !mo.tensor<[3], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu_mixed_types() -> !mo.tensor<[3], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<3, 0, 5> : !M.array<3xsi32>> : !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[3], si32>
mo.graph @support_convert_relu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant <#M.dense_array<3, 0, 5> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %1 = "relu"(%0) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], si32>
  mo.output %1 : !mo.tensor<[3], si32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu6() -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<0.000000e+00, 6.000000e+00, 0.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<5xf32>> : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[5], f32>
mo.graph @support_convert_relu6() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-2.0, 18.0, 0.0, 5.0, 6.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = "relu6"(%0) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu6_mixed_types() -> !mo.tensor<[2], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<6, 6> : !M.array<2xsi32>> : !mo.tensor<[2], si32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[2], si32>
mo.graph @support_convert_relu6_mixed_types() -> (!mo.tensor<[2], si32>) {
  %0 = mo.constant <#M.dense_array<7, 8> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %1 = "relu6"(%0) : (!mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>
  mo.output %1 : !mo.tensor<[2], si32>
}
