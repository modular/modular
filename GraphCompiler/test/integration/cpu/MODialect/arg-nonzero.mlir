// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones --input-specs=2x2x3xf32 %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'arg_nonzero_op':
// CHECK-NEXT: 'arg_nonzero_op' returned tensor<4x1xsi64> [1, 2, 3, 4]
mo.graph @arg_nonzero_op<D0, D1, D2 -> D3>(%dummy: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, 1], si64> {
  %val1 = mo.constant <#M.dense_array<0.0, 1.0, -1.0, 2.0, 1.0, 0.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %0 = mo.arg_nonzero<() -> D4>(%val1) : (!mo.tensor<[6], f32>) -> !mo.tensor<[D4, 1], si64>
  mo.output<D4> %0 : !mo.tensor<[D4, 1], si64>
}

// CHECK-LABEL: --- Running 'arg_nonzero_dyn':
// NOTE: group each entry by 3 [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), ...
// CHECK-NEXT: 'arg_nonzero_dyn' returned tensor<12x3xsi64> [0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2]
mo.graph @arg_nonzero_dyn<D0, D1, D2 -> D3>(%arg0: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D3, 3], si64> {
  %0 = mo.arg_nonzero<() -> D4>(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D4, 3], si64>
  mo.output<D4> %0 : !mo.tensor<[D4, 3], si64>
}
