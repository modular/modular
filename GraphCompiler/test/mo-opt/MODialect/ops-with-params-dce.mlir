// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s


// CHECK: mo.graph @ops_with_params_no_dce<[[D0:[^,>]*]]>(%[[ARG0:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, %[[SH:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<{{\[}}[[D0]]{{\]}}, si32> {
mo.graph @ops_with_params_no_dce<D0>(%arg0: !mo.tensor<[D0], si32>, %sh: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si32> {
  // Operations with parameters that are not used get DCE'd, but their parameter declarations remain
  // CHECK-DAG: kgen.param.declare [[D1:[^ ]*]]: scalar<si64> = <[[D0]]>
  %0 = mo.rebind<() -> D1>(%arg0) : !mo.tensor<[D0], si32> -> !mo.tensor<[D1], si32>
  // CHECK-DAG: %{{.*}} = mo.broadcast_to<() -> [[D4:.*]], [[D5:.*]]>
  %1 = mo.broadcast_to<() -> D4, D5>(%arg0, %sh) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[2], si64>)-> !mo.tensor<[D4, D5], si32>
  // CHECK-DAG: %{{.*}} = mo.reshape<() -> [[D2:.*]], [[D3:.*]]>
  %2 = mo.reshape<() -> D2, D3>(%arg0, %arg0) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[D0], si32>)-> !mo.tensor<[D2, D3], si32>
  // CHECK: mo.output %[[ARG0]] : !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  mo.output %arg0 : !mo.tensor<[D0], si32>
}

// CHECK: mo.graph @ops_without_params_dce<[[D0:[^,>]*]]>(%[[ARG0:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, %[[SH:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<{{\[}}[[D0]]{{\]}}, si32> {
mo.graph @ops_without_params_dce<D0>(%arg0: !mo.tensor<[D0], si32>, %sh: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si32> {
  // Operations without result parameters get fully DCE'd
  %0 = mo.rebind(%arg0) : !mo.tensor<[D0], si32> -> !mo.tensor<[D0], si32>
  %1 = mo.broadcast_to(%arg0, %sh) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[2], si64>
  )-> !mo.tensor<[?, ?], si32>
  %2 = mo.reshape(%arg0, %arg0) : (
    !mo.tensor<[D0], si32>, !mo.tensor<[D0], si32>
  )-> !mo.tensor<[D0], si32>
  // CHECK: mo.output %[[ARG0]] : !mo.tensor<{{\[}}[[D0]]{{\]}}, si32>
  mo.output %arg0 : !mo.tensor<[D0], si32>
}
