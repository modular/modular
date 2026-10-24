// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @conv2d(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 16], f32>,
// CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> {
mo.graph @conv2d(
  %input: !mo.tensor<[10, 5, 5, 16], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[10, 2, 2, 32], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[NG:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  // CHECK: mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], num_groups = %[[NG]], filter_layout = ""]
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[10, 2, 2, 32], f32>
  mo.output %0 : !mo.tensor<[10, 2, 2, 32], f32>
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @conv_transpose2d(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 2, 2, 32], f32>,
// CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 4, 4, 16], f32> {
mo.graph @conv_transpose2d(
  %input: !mo.tensor<[10, 2, 2, 32], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[10, 4, 4, 16], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[OP:.*]] = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: mo.conv_transpose(%[[INPUT]], %[[FILTER]])
  // CHECK-SAME: [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], output_paddings = %[[OP]], filter_layout = ""]
  %0 = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[10, 2, 2, 32], f32>, !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[10, 4, 4, 16], f32>
  mo.output %0 : !mo.tensor<[10, 4, 4, 16], f32>
}

//===----------------------------------------------------------------------===//
// AvgPoolOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @avg_pool(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
mo.graph @avg_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK: mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], count_boundary = false]
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @max_pool(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
mo.graph @max_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK: mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]]
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

