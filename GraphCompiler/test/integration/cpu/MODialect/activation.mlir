// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @activation_relu<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'activation_relu' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @activation_sigmoid<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.sigmoid(%arg0) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'activation_sigmoid' returned tensor<5xf32> [7.310586e-01, 7.310586e-01, 7.310586e-01, 7.310586e-01, 7.310586e-01]
