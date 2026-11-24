// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=1x5x5x3 --input-shapes=3x3x3x1 --result-output-style=full %t.mlir | FileCheck %s

mo.graph @conv<D0, D1, D2, D3, D4, D5, D6, D7 -> D8, D9, D10, D11>(%input : !mo.tensor<[D0, D1, D2, D3], f32>, %filter : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[D8, D9, D10, D11], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv<() -> D12, D13, D14, D15>(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D12, D13, D14, D15], f32>
  mo.output<D12, D13, D14, D15> %output : !mo.tensor<[D12, D13, D14, D15], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'conv' returned tensor<1x3x3x1xf32> [5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01]

mo.graph @strided_conv<D0, D1, D2, D3, D4, D5, D6, D7 -> D8, D9, D10, D11>(%input : !mo.tensor<[D0, D1, D2, D3], f32>, %filter : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[D8, D9, D10, D11], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"(){value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv<() -> D12, D13, D14, D15>(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D12, D13, D14, D15], f32>
  mo.output<D12, D13, D14, D15> %output : !mo.tensor<[D12, D13, D14, D15], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'strided_conv' returned tensor<1x2x2x1xf32> [5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01]

mo.graph @padded_conv<D0, D1, D2, D3, D4, D5, D6, D7 -> D8, D9, D10, D11>(%input : !mo.tensor<[D0, D1, D2, D3], f32>, %filter : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[D8, D9, D10, D11], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<1, 1, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"(){value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv<() -> D12, D13, D14, D15>(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6, D7], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D12, D13, D14, D15], f32>
  mo.output<D12, D13, D14, D15> %output : !mo.tensor<[D12, D13, D14, D15], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'padded_conv' returned tensor<1x5x3x1xf32> [3.600000e+01, 3.600000e+01, 3.600000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01]

mo.graph @grouped_conv<D0, D1, D2, D3 -> D4, D5, D6, D7>(%input : !mo.tensor<[D0, D1, D2, D3], f32>) -> (!mo.tensor<[D4, D5, D6, D7], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %filter = "mo.constant"() {value = #M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<3x3x1x3xf32>} : () -> !mo.tensor<[3, 3, 1, 3], f32, {layout = #mo.layout<RSCF>}>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 1, 3], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv<() -> D8, D9, D10, D11>(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[3, 3, 1, 3], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %output : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'grouped_conv' returned tensor<1x3x3x3xf32> [1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01]
