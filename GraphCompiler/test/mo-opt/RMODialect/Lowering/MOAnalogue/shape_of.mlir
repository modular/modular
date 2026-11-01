// RUN: mo-opt %s --mo-legalize-rmo-operators="skip-rmo-rebinds" | FileCheck %s

mo.graph @shape_of_params<D0, D1, D2 -> D3>(%input: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3], si64> {
    %1 = rmo.mo.shape_of<() -> D4>(%input): (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D4], si64>
    mo.output<D4> %1: !mo.tensor<[D4], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_params<D0, D1, D2 -> D3>(%[[ARG0:.*]]: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3], si64> {
// CHECK-NEXT: %[[V0:.*]] = mo.shape_of(%[[ARG0]]) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D4:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[3], si64>) -> !mo.tensor<[[[D4]]], si64>
// CHECK-NEXT: mo.output<[[D4]]> %[[V1]] : !mo.tensor<[[[D4]]], si64>
