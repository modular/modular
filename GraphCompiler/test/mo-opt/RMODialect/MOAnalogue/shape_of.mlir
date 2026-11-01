// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @shape_of_ok<D0, D1>(%input: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64> {
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64>
    mo.output %1: !mo.tensor<[2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_ok<D0, D1>([[VARG0:[^ ]*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.shape_of([[VARG0]]) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[2], si64>

// -----

mo.graph @shape_of_not_ok<D0, D1>(%input: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], f32> {
    // expected-error@below {{'rmo.mo.shape_of' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], f32>
    mo.output %1: !mo.tensor<[2], f32>
}
