// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x3 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @cumsumDynamic<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %output = mo.cumsum(%arg0) [axis = 1, exclusive = 0, reverse = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %output : !mo.tensor<[D0, D1], f32>
}
// CHECK{LITERAL}: 'cumsumDynamic' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]
