// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @broadcast_to_ok<D0 -> D1>(%data: !mo.tensor<[D0], si64>, %shape: !mo.tensor<[1], si64>) -> !mo.tensor<[D1], si64> {
    %1 = rmo.mo.broadcast_to<() -> D2>(%data, %shape) : (!mo.tensor<[D0], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[D2], si64>
    mo.output<D2> %1: !mo.tensor<[D2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_ok<D0 -> D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1], si64>) -> !mo.tensor<[D1], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_to<() -> D2>([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[D2], si64>
// CHECK-NEXT: mo.output<D2> [[V0]] : !mo.tensor<[D2], si64>

// -----

mo.graph @broadcast_to_ok_f32<D0 -> D1>(%data: !mo.tensor<[D0], f32>, %shape: !mo.tensor<[1], si64>) -> !mo.tensor<[D1], f32> {
    %1 = rmo.mo.broadcast_to<() -> D2>(%data, %shape) : (!mo.tensor<[D0], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D2], f32>
    mo.output<D2> %1: !mo.tensor<[D2], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_ok_f32<D0 -> D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1], si64>) -> !mo.tensor<[D1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_to<() -> D2>([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D2], f32>
// CHECK-NEXT: mo.output<D2> [[V0]] : !mo.tensor<[D2], f32>

// -----

mo.graph @broadcast_to_not_ok<D0, D1, D2>(%data: !mo.tensor<[D0], si64>, %shape: !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64> {
    // expected-error@below {{'rmo.mo.broadcast_to' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D1, D2], f32>'}}
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64>
    mo.output %1: !mo.tensor<[D0], si64>
}
