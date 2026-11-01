// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=5 --input-shapes=5 --input-shapes=5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @select<D0>(%arg0 : !mo.tensor<[D0], bool>, %arg1 : !mo.tensor<[D0], f32>, %arg2 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %1 = mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[D0], bool>, !mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32>
  mo.output %1 : !mo.tensor<[D0], f32>
}
// CHECK: 'select' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
