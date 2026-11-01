// RUN: mo-opt %s | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @batch_matmul(%[[ARG0:.*]]: !mo.tensor<[3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> {
mo.graph @batch_matmul(%arg0: !mo.tensor<[3, 10, 20], f32>, %arg1: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  mo.output %0 : !mo.tensor<[3, 10, 5], f32>
}

// CHECK: mo.graph @batch_matmul_4D<D0, D1>(%[[ARG0:.*]]: !mo.tensor<[7, 3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32> {
mo.graph @batch_matmul_4D<D0, D1>(%arg0: !mo.tensor<[7, 3, 10, 20], f32>, %arg1: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32>
  mo.output %0 : !mo.tensor<[D0, D1, 10, 5], f32>
}

// CHECK: mo.graph @batch_matmul_mixed_precision(%[[ARG0:.*]]: !mo.tensor<[3, 10, 20], ui8>, %[[ARG1:.*]]: !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32> {
mo.graph @batch_matmul_mixed_precision(%arg0: !mo.tensor<[3, 10, 20], ui8>, %arg1: !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 10, 20], ui8>, !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[3, 10, 20], ui8>, !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32>
  mo.output %0 : !mo.tensor<[3, 10, 5], si32>
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @matmul(%[[ARG0:.*]]: !mo.tensor<[10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
mo.graph @matmul(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_dynamic<D0, D1, D2, D3, D4, D5>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f32> {
mo.graph @matmul_dynamic<D0, D1, D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>) -> (!mo.tensor<[D4, D5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)   -> !mo.tensor<[D4, D5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)   -> !mo.tensor<[D4, D5], f32>
  mo.output %0 : !mo.tensor<[D4, D5], f32>
}

// CHECK: mo.graph @matmul_dyndim_1<D0, D1>
mo.graph @matmul_dyndim_1<D0, D1>(%arg0: !mo.tensor<[D0, 10], f32>, %arg1: !mo.tensor<[10, 5], f32>) -> (!mo.tensor<[D1, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[D1, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[D0, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[D1, 5], f32>
  mo.output %0 : !mo.tensor<[D1, 5], f32>
}

// CHECK: mo.graph @matmul_dyndim_2<D0, D1>
mo.graph @matmul_dyndim_2<D0, D1>(%arg0: !mo.tensor<[10, 10], f32>, %arg1: !mo.tensor<[10, D0], f32>) -> (!mo.tensor<[10, D1], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, D0], f32>) -> !mo.tensor<[10, D1], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, D0], f32>) -> !mo.tensor<[10, D1], f32>
  mo.output %0 : !mo.tensor<[10, D1], f32>
}

// CHECK: mo.graph @matmul_dyndim_K<D0, D1>
mo.graph @matmul_dyndim_K<D0, D1>(%arg0: !mo.tensor<[10, D0], f32>, %arg1: !mo.tensor<[D1, 5], f32>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 5], f32>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_fp16_mixed_precision<D0, D1>
mo.graph @matmul_fp16_mixed_precision<D0, D1>(%arg0: !mo.tensor<[10, D0], f16>, %arg1: !mo.tensor<[D1, 5], f16>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], f16>, !mo.tensor<[D1, 5], f16>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], f16>, !mo.tensor<[D1, 5], f16>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_si8_ui8_si32_mixed_precision<D0, D1>
mo.graph @matmul_si8_ui8_si32_mixed_precision<D0, D1>(%arg0: !mo.tensor<[10, D0], si8>, %arg1: !mo.tensor<[D1, 5], ui8>) -> (!mo.tensor<[10, 5], si32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], si8>, !mo.tensor<[D1, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], si8>, !mo.tensor<[D1, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  mo.output %0 : !mo.tensor<[10, 5], si32>
}

