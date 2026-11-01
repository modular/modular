// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=5 --input-shapes=5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @equal<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool> {
  %0 = mo.equal(%arg0, %arg1) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'equal' returned tensor<5xbool> [True, True, True, True, True]

mo.graph @greater<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool> {
  %0 = mo.greater(%arg0, %arg1) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'greater' returned tensor<5xbool> [False, False, False, False, False]

mo.graph @greater_equal<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool> {
  %0 = mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'greater_equal' returned tensor<5xbool> [True, True, True, True, True]

mo.graph @not_equal<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool> {
  %0 = mo.not_equal(%arg0, %arg1) : (!mo.tensor<[D0], f32>, !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'not_equal' returned tensor<5xbool> [False, False, False, False, False]
