// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @broadcast_shape_ok<D0, D1 -> D2>(%shape0: !mo.tensor<[D0], si64>, %shape1: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D3>(%shape0, %shape1) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>
    mo.output<D3> %1: !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_ok<D0, D1 -> D2>([[VARG0:[^ ]*]]: !mo.tensor<[D0], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_shape<() -> D3>([[VARG0]], [[VARG1]]) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>
// CHECK-NEXT: mo.output<D3> [[V0]] : !mo.tensor<[D3], si64>

// -----

mo.graph @broadcast_shape_not_ok<D0, D1, D2>(%shape0: !mo.tensor<[D0], si64>, %shape1: !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64> {
    // expected-error@below {{'rmo.mo.broadcast_shape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D1, D2], f32>'}}
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64>
    mo.output %1: !mo.tensor<[D0], si64>
}
