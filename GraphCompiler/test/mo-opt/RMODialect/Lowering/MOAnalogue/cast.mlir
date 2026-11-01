// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @cast_params<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, D4, D5], si32> {
  %1 = rmo.mo.cast<() -> D6, D7, D8>(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D6, D7, D8], si32>
  mo.output<D6, D7, D8> %1 : !mo.tensor<[D6, D7, D8], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_params<D0, D1, D2 -> D3, D4, D5>(%[[ARG0:.*]]: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, D4, D5], si32> {
// CHECK-NEXT: %[[V0:.*]] = mo.cast(%[[ARG0]]) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D0, D1, D2], si32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1, D2], si32>) -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>
// CHECK-NEXT: mo.output<[[D6]], [[D7]], [[D8]]> %[[V1]] : !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>

// -----

mo.graph @cast_static<() -> D0, D1, D2>(%arg0: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[D0, D1, D2], si32> {
  %1 = rmo.mo.cast<() -> D3, D4, D5>(%arg0) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[D3, D4, D5], si32>
  mo.output<D3, D4, D5> %1 : !mo.tensor<[D3, D4, D5], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_static<() -> D0, D1, D2>(%[[ARG0:.*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[D0, D1, D2], si32> {
// CHECK-NEXT: %[[V0:.*]] = mo.cast(%[[ARG0]]) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], si32>
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[V1]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], si32>
