// RUN: mo-opt %s --canonicalize | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @shape_of_fold_static
mo.graph @shape_of_fold_static(%arg0: !mo.tensor<[1, 1, 4, 14], bf16>) -> !mo.tensor<[4], si64> {
  // CHECK-NEXT: [[V0:%.*]] = mo.constant <#M.dense_array<1, 1, 4, 14> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %0 = mo.shape_of(%arg0) : (!mo.tensor<[1, 1, 4, 14], bf16>) -> !mo.tensor<[4], si64>
  // CHECK-NEXT: mo.output [[V0]]
  mo.output %0 : !mo.tensor<[4], si64>
}

// CHECK-LABEL: mo.graph @shape_of_fold_dim_parameter<D0>
mo.graph @shape_of_fold_dim_parameter<D0>(%arg0: !mo.tensor<[D0, 1], f32>) -> !mo.tensor<[2], si64> {
  // CHECK-NEXT: [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[D0, 1]>
  // CHECK-NEXT: [[V1:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[2], si64>
  %0 = mo.shape_of(%arg0) : (!mo.tensor<[D0, 1], f32>) -> !mo.tensor<[2], si64>
  // CHECK-NEXT: mo.output [[V1]]
  mo.output %0 : !mo.tensor<[2], si64>
}

// CHECK-LABEL: mo.graph @shape_of_no_fold_1
// CHECK-SAME: <[[D0:[^,>]*]]>(%{{.*}}: !mo.tensor<{{\[}}[[D0]], 1{{\]}}, si32>) -> !mo.tensor<[2], si64>
mo.graph @shape_of_no_fold_1<D0>(%arg0: !mo.tensor<[D0, 1], si32>) -> !mo.tensor<[2], si64> {
  // CHECK: mosh.param.to_value
  // CHECK: mo.shape.to_tensor
  %0 = mo.shape_of(%arg0) : (!mo.tensor<[D0, 1], si32>) -> !mo.tensor<[2], si64>
  mo.output %0 : !mo.tensor<[2], si64>
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @to_tensor_fold
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[SH0:.*]]: !mosh.ape>() -> (
mo.graph @to_tensor_fold<D0, D1, Sh0: !mosh.ape>() -> (
  // CHECK-SAME: !mo.tensor<{{\[}}[[D1]]{{\]}}, si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  !mo.tensor<[D1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[D0], si32>
) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant <#M.dense_array<17, 42> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH0]]>
  // CHECK-DAG: %[[VAL2:.*]] = mosh.param.to_value: !mosh.ape = <{{\[}}[[D0]], 42{{\]}}>
  // CHECK-DAG: %[[VAL3:.*]] = mo.shape.to_tensor(%[[VAL1]]) -> !mo.tensor<{{\[}}[[D1]]{{\]}}, si64>
  // CHECK-DAG: %[[VAL4:.*]] = mo.shape.to_tensor(%[[VAL2]]) -> !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.constant <#M.dense_array<23> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  // CHECK-DAG: %[[VAL6:.*]] = mo.rebind(%[[VAL5]]) : !mo.tensor<[1], si32> -> !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  %sh0 = mosh.param.to_value: !mosh.ape = <Sh0>
  %sh1 = mosh.param.to_value: !mosh.ape = <[D0, 42]>
  %sh2 = mosh.param.to_value: !mosh.ape = <[17, 42]>
  %sh3 = mosh.param.to_value: !mosh.ape = <[23]>
  %0 = mo.shape.to_tensor(%sh0) -> !mo.tensor<[D1], si64>
  %1 = mo.shape.to_tensor(%sh1) -> !mo.tensor<[2], si64>
  %2 = mo.shape.to_tensor(%sh2) -> !mo.tensor<[2], si64>
  %3 = mo.shape.to_tensor(%sh3) -> !mo.tensor<[1], si32>
  %4 = mo.shape.to_tensor(%sh3) -> !mo.tensor<[D0], si32>
  // CHECK-NEXT: mo.output %[[VAL3]], %[[VAL4]], %[[VAL0]], %[[VAL5]], %[[VAL6]] : !mo.tensor<{{\[}}[[D1]]{{\]}}, si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  mo.output %0, %1, %2, %3, %4 : !mo.tensor<[D1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[D0], si32>
}

// CHECK: mo.graph @to_tensor_fold_unknown<[[K:.*]], [[D0:.*]]>() -> !mo.tensor<{{\[}}[[D0]]{{\]}}, si32> {
mo.graph @to_tensor_fold_unknown<K, D0>() -> !mo.tensor<[D0], si32> {
  // CHECK: %[[VAL0:.*]] = mosh.param.to_value: !mosh.ape = <{{\[}}[[K]]{{\]}}>
  %0 = mosh.param.to_value: !mosh.ape = <[K]>
  // CHECK: %[[VAL1:.*]] = mo.shape.to_tensor(%[[VAL0]]) -> !mo.tensor<[1], si32>
  %1 = mo.shape.to_tensor(%0) -> !mo.tensor<[D0], si32>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[1], si32> -> !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  mo.output %1 : !mo.tensor<[D0], si32>
}

// CHECK-LABEL: mo.graph @to_tensor_from_tensor
// CHECK:       (%[[ARG:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
mo.graph @to_tensor_from_tensor(%arg: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64> {
  %sh = mo.shape.from_tensor(%arg) : (!mo.tensor<[4], si64>) -> !mosh.ape
  %tensor = mo.shape.to_tensor(%sh) -> !mo.tensor<[4], si64>

  // CHECK: mo.output %[[ARG]]
  mo.output %tensor : !mo.tensor<[4], si64>
}

// CHECK-LABEL: mo.graph @to_tensor_from_tensor_type_mismatch
// CHECK:       (%[[ARG:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si32>
mo.graph @to_tensor_from_tensor_type_mismatch(%arg: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si32> {

  // CHECK-DAG: %[[SH:.*]] = mo.shape.from_tensor
  // CHECK-DAG: %[[OUT:.*]] = mo.shape.to_tensor(%[[SH]])
  %sh = mo.shape.from_tensor(%arg) : (!mo.tensor<[4], si64>) -> !mosh.ape
  %tensor = mo.shape.to_tensor(%sh) -> !mo.tensor<[4], si32>

  // CHECK: mo.output %[[OUT]]
  mo.output %tensor : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// FromTensorOp
//===----------------------------------------------------------------------===//

mo.graph @from_tensor_to_tensor<Sh: !mosh.ape -> O0: !mosh.ape>(%arg: !mo.tensor<[4], si64>) {
  %sh = mosh.param.to_value : !mosh.ape = <Sh>

  %tensor = mo.shape.to_tensor(%sh) -> !mo.tensor<[4], si64>
  %sh2 = mo.shape.from_tensor(%tensor) : (!mo.tensor<[4], si64>) -> !mosh.ape

  mosh.param.from_value Out : !mosh.ape = %sh2
  mo.output<:!mosh.ape Out>
}

// CHECK-LABEL: mo.graph @from_tensor_constant
// CHECK: kgen.param.declare SH: !mosh.ape = <[2]>
mo.graph @from_tensor_constant<() -> Sh0: !mosh.ape>() {
  %1 = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[1], si64>
  %sh = mo.shape.from_tensor(%1) : (!mo.tensor<[1], si64>) -> !mosh.ape

  mosh.param.from_value SH : !mosh.ape = %sh
  mo.output<:!mosh.ape SH>
}

