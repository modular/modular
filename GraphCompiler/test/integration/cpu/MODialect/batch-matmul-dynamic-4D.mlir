// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=1x2x4x2 --input-shapes=1x2x2x3 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @batchMatmulDynamic<D0, D1, D2, D3, D4>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[D0, D1, D3, D4], f32>) -> !mo.tensor<[D0, D1, D2, D4], f32> {
  %output = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D0, D1, D3, D4], f32>) -> !mo.tensor<[D0, D1, D2, D4], f32>
  mo.output %output : !mo.tensor<[D0, D1, D2, D4], f32>
}
// CHECK{LITERAL}: 'batchMatmulDynamic' returned tensor<1x2x4x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
