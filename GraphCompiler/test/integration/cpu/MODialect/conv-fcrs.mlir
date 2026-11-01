// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// FCRS filters are transposed to RSCF at compile time before packing.
// This test disables constant folding, so the transpose runs at runtime.

// CHECK{LITERAL}: 'conv' returned tensor<1x3x3x1xf32> [5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01]
mo.graph @conv(%input : !mo.tensor<[1, 5, 5, 3], f32>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0> : tensor<1x3x3x3xf32>} : () -> !mo.tensor<[1, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>
  %output = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[1, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
