// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @bcast_like_ok<D0, D1, D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f64>) -> !mo.tensor<[D4, D5], f32> {
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f64>) -> !mo.tensor<[D4, D5], f32>
  mo.output %1 : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_ok<[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], f64>) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.broadcast_like([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f64>) -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: }

// -----

mo.graph @bcast_like_not_ok_bad_operands<D0, D1, D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f64>, %arg1: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f32> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<[D0, D1], f64>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output %1 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @bcast_like_not_ok_bad_results<D0, D1, D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f64> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f64>
  mo.output %1 : !mo.tensor<[D4, D5], f64>
}
