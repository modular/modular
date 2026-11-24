// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @broadcast_shape_params_params<D0, D1 -> D2>(%shape0: !mo.tensor<[D0], si64>, %shape1: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D3>(%shape0, %shape1) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>
    mo.output<D3> %1: !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_params_params<D0, D1 -> D2>(%[[ARG0:.*]]: !mo.tensor<[D0], si64>, %[[ARG1:.*]]: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape<() -> [[D3:.*]]>(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[[[D3]]], si64>
// CHECK-NEXT: mo.output<[[D3]]> %[[V0]] : !mo.tensor<[[[D3]]], si64>

// -----

mo.graph @broadcast_shape_static_static<() -> D0>(%shape0: !mo.tensor<[3], si64>, %shape1: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D1>(%shape0, %shape1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_static_static<() -> D0>(%[[ARG0:.*]]: !mo.tensor<[3], si64>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si64> {
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape<() -> [[D1:.*]]>(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> %[[V0]] : !mo.tensor<[[[D1]]], si64>
