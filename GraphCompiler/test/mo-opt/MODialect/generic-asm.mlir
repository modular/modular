// RUN: mo-opt --mlir-print-op-generic %s | mo-opt | FileCheck %s

// CHECK-LABEL: mo.graph @materialize_linalg_solve
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]]>(
mo.graph @materialize_linalg_solve<D0, D1 -> D2>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, %[[ARG1:.*]]: !mo.tensor<[5, [[D1]]], f32>
  %arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[5, D1], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D2]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>) {
) -> (!mo.tensor<[D2, D1], f32>, !mo.tensor<[D0, D1], f32>) {
  // CHECK-NEXT: mo.output<5> %[[ARG1]], %[[ARG0]] : !mo.tensor<[5, [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>
  mo.output<5> %arg1, %arg0 : !mo.tensor<[5, D1], f32>, !mo.tensor<[D0, D1], f32>
}
