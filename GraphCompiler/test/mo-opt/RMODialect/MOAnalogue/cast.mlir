// RUN: mo-opt %s | FileCheck %s

mo.graph @cast_ok<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, D4, D5], si32> {
  %1 = rmo.mo.cast<() -> D6, D7, D8>(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D6, D7, D8], si32>
  mo.output<D6, D7, D8> %1 : !mo.tensor<[D6, D7, D8], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_ok<D0, D1, D2 -> D3, D4, D5>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, D4, D5], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.cast<() -> D6, D7, D8>([[VARG0]]) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D6, D7, D8], si32>
// CHECK-NEXT: mo.output<D6, D7, D8> [[V0]] : !mo.tensor<[D6, D7, D8], si32>
