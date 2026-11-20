// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators %s | FileCheck %s --enable-var-scope

mo.graph @shape_to_tensor() -> !mo.tensor<[1], si64> {
  %1 = rmo.shape_to_tensor() [shape = #mosh<ape[7]> : !mosh.ape] : () -> !mo.tensor<[1], si64>
  mo.output %1 : !mo.tensor<[1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_to_tensor() -> !mo.tensor<[1], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mosh.param.to_value: !mosh.ape = <[7]>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[1], si64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1], si64>

// -----

mo.graph @shape_to_tensor_many() -> !mo.tensor<[3], si64> {
  %1 = rmo.shape_to_tensor() [shape = #mosh<ape[7, 77, 22]> : !mosh.ape] : () -> !mo.tensor<[3], si64>
  mo.output %1 : !mo.tensor<[3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_to_tensor_many() -> !mo.tensor<[3], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mosh.param.to_value: !mosh.ape = <[7, 77, 22]>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[3], si64>

// -----

mo.graph @shape_to_tensor_symbolic<batch, seqlen>() -> !mo.tensor<[3], si64> {
  %1 = rmo.shape_to_tensor() [shape = #mosh<ape[batch, seqlen, 22]> : !mosh.ape] : () -> !mo.tensor<[3], si64>
  mo.output %1 : !mo.tensor<[3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_to_tensor_symbolic<batch, seqlen>() -> !mo.tensor<[3], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mosh.param.to_value: !mosh.ape = <[batch, seqlen, 22]>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[3], si64>

// -----

// Device must be provided by user so we cannot fail type inference if we
// cannot predict it.
mo.graph @shape_to_tensor_symbolic_different_device<batch, seqlen>() -> !mo.tensor<[3], si64> {
  %1 = rmo.shape_to_tensor() [shape = #mosh<ape[batch, seqlen, 22]> : !mosh.ape] : () -> !mo.tensor<[3], si64>
  mo.output %1 : !mo.tensor<[3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_to_tensor_symbolic_different_device<batch, seqlen>() -> !mo.tensor<[3], si64>  {
// CHECK-NEXT: [[V0:[^ ]*]] = mosh.param.to_value: !mosh.ape = <[batch, seqlen, 22]>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[3], si64>
