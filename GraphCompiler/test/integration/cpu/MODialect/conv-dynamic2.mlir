// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=11x4x6x5 --input-shapes=2x3x5x7 --result-output-style=full %t.mlir | FileCheck %s

mo.graph @conv_dynamic_batch<D0>(%arg0: !mo.tensor<[D0, 4, 6, 5], f32>, %arg1: !mo.tensor<[2, 3, 5, 7], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[D0, 3, 4, 7], f32> {
    %0 = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
    %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %1 = mo.conv(%arg0, %arg1) [strides = %0, dilations = %0, paddings = %pa, num_groups = %ng, filter_layout = ""] : (!mo.tensor<[D0, 4, 6, 5], f32>, !mo.tensor<[2, 3, 5, 7], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, 3, 4, 7], f32>
    mo.output %1 : !mo.tensor<[D0, 3, 4, 7], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK: 'conv_dynamic_batch' returned tensor<11x3x4x7xf32> [3.000000e+01, {{.*}}]
