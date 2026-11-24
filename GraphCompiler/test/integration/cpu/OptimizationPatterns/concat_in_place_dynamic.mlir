// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=3 --result-output-style=full %t.mlir | FileCheck %s

// CHECK{LITERAL}: 'concat_dynamic_output' returned tensor<6xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 7.615942e-01, 7.615942e-01, 7.615942e-01]

mo.graph @concat_dynamic_output<D0>(%arg0: !mo.tensor<[D0], f32>) -> !mo.tensor<[add(D0, D0)], f32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %0 = mo.relu(%arg0) : !mo.tensor<[D0], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[D0], f32>
  %out = mo.concat[0](%0, %1) : (
    !mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>
  ) -> !mo.tensor<[add(D0, D0)], f32>
  mo.output %out : !mo.tensor<[add(D0, D0)], f32>
}
