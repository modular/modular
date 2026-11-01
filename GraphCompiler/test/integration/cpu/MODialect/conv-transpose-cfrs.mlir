// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// TODO: Support Conv FCSR support on CPU (GEX-2302)
// XFAIL: *

mo.graph @conv_transpose(%input: !mo.tensor<[1, 2, 2, 16], f32>)
                         -> (!mo.tensor<[1, 3, 3, 8], f32>) {
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si64>
  %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<f32>} : () -> !mo.tensor<[16, 8, 2, 2], f32, {layout = #mo.layout<CFRS>}>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 16], f32>, !mo.tensor<[16, 8, 2, 2], f32, {layout = #mo.layout<CFRS>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 3, 3, 8], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 8], f32>
}
