// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.reduce.softmax(%[[ARG0]]) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.softmax(%arg0) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.reduce.logsoftmax(%[[ARG0]]) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.logsoftmax(%arg0) [axis = 1] : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
