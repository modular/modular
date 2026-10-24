// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


mo.graph @range_folded_away<() -> D0>() -> (!mo.tensor<[D0], ui32>) {
  %start = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
  %limit = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
  %step = mo.constant <#M.dense_array<2> : tensor<ui32>> : !mo.tensor<[], ui32>
  %0 = rmo.mo.range<() -> D1>(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[D1], ui32>

  mo.output<D1> %0 : !mo.tensor<[D1], ui32>
}
// CHECK-LABEL: mo.graph @range_folded_away
// CHECK-SAME: <() -> D0>() -> !mo.tensor<[D0], ui32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>> : !mo.tensor<[5], ui32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D1:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[5], ui32>) -> !mo.tensor<[[[D1]]], ui32>
// CHECK-NEXT: mo.output<[[D1]]> [[V1]] : !mo.tensor<[[[D1]]], ui32>

// Note: range_folded_away keeps the rebind because the inferMOTypes fast-path
// for constant inputs returns a fully-static shape (`[5]`) rather than the
// user's declared `D1` — a rebind is genuinely needed.

// -----

mo.graph @non_constant_range_input<() -> D0>(%step: !mo.tensor<[], ui32>) -> (!mo.tensor<[D0], ui32>) {
  %start = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
  %limit = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
  %0 = rmo.mo.range<() -> D1>(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[D1], ui32>

  mo.output<D1> %0 : !mo.tensor<[D1], ui32>
}
// CHECK-LABEL: mo.graph @non_constant_range_input
// CHECK-SAME: <() -> D0>(%[[ARG0:[^ ]*]]: !mo.tensor<[], ui32>) -> !mo.tensor<[D0], ui32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.range<() -> [[D1:[^ ]*]]>([[V0]], [[V1]], %[[ARG0]]) : (!mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[[[D1]]], ui32>
// CHECK-NEXT: mo.output<[[D1]]> [[V2]] : !mo.tensor<[[[D1]]], ui32>

// -----


mo.graph @range_floats() -> (!mo.tensor<[5], f32>) {
  %start = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %limit = mo.constant <#M.dense_array<11.0> : tensor<f32>> : !mo.tensor<[], f32>
  %step = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %0 = rmo.mo.range(%start, %limit, %step) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>

  mo.output %0 : !mo.tensor<[5], f32>
}

// CHECK-LABEL: mo.graph @range_floats
// CHECK-SAME: () -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<1.100000e+01> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.range([[V0]], [[V1]], [[V2]]) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[5], f32>

// -----

mo.graph @range_floats_unknown_output_shape<() -> D0>() -> (!mo.tensor<[D0], f32>) {
  %start = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %limit = mo.constant <#M.dense_array<11.0> : tensor<f32>> : !mo.tensor<[], f32>
  %step = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %0 = rmo.mo.range<() -> D1>(%start, %limit, %step) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D1], f32>

  mo.output<D1> %0 : !mo.tensor<[D1], f32>
}
// CHECK-LABEL: mo.graph @range_floats_unknown_output_shape
// CHECK-SAME: <() -> D0>() -> !mo.tensor<[D0], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<1.100000e+01> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.range([[V0]], [[V1]], [[V2]]) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D1:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[5], f32>) -> !mo.tensor<[[[D1]]], f32>
// CHECK-NEXT: mo.output<[[D1]]> [[V4]] : !mo.tensor<[[[D1]]], f32>

// Note: range_floats_unknown_output_shape keeps the rebind for the same
// reason as range_folded_away — the constant-input fast-path returns a
// fully-static `[5]` rather than the user's declared `D1`.
