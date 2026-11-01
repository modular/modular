// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @split_zero_axis(
  %input: !mo.tensor<[5, 4], f32>
) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>) {
  %split_sizes = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res:2 = mo.split(%input, %split_sizes) [axis = 0] : (
    !mo.tensor<[5, 4], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>
}

// CHECK: 'split_zero_axis' returned tensor<2x4xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
// CHECK-NEXT: , tensor<3x4xf32>  [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
