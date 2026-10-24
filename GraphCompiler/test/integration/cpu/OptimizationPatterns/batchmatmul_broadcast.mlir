// RUN: mo-opt -mo-shape-inference -mo-symbolic-optimize %s | FileCheck %s --check-prefix=CHECK-IR
// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s --check-prefix=CHECK-EXEC

// CHECK-IR: mo.graph @bmm_broadcast(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>
mo.graph @bmm_broadcast(%arg0 : !mo.tensor<[4, 3, 2], f32>, %arg1 : !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4, 3, 3], f32>) {
  %c = "mo.constant"() {value = #M.dense_array<4, 2, 3> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  // CHECK-IR: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]
  %1 = mo.broadcast_to(%arg1, %c) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.matmul(%[[VAL0]], %[[ARG1]])
  %2 = mo.batch_matmul(%arg0, %1) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>
  // CHECK-IR: %[[OUT:.*]] = mo.static.reshape(%[[VAL1]]
  // CHECK-IR: mo.output %[[OUT]]
  mo.output %2 : !mo.tensor<[4, 3, 3], f32>
}
// CHECK-EXEC: 'bmm_broadcast' returned tensor<4x3x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00
