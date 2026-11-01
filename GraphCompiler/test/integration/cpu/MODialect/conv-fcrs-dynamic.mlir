// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=1x5x5x3 --input-shapes=1x3x3x3 --result-output-style=full %t.mlir | FileCheck %s

// TODO: Support Conv FCSR support on CPU (GEX-2302)
// XFAIL: *

mo.graph @conv(%input : !mo.tensor<[?, ?, ?, ?], f32>, %filter : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>) -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %output = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?, ?], f32>
}
