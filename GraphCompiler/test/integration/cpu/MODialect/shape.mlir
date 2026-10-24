// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=1x2x3 --input-data=0::ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @shape_of<D0, D1, D2>(%arg0 : !mo.tensor<[D0, D1, D2], f32>) -> (!mo.tensor<[3], si64>) {
  %output = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si64>
  mo.output %output : !mo.tensor<[3], si64>
}
// CHECK{LITERAL}: 'shape_of' returned tensor<3xsi64> [1, 2, 3]

mo.graph @shape_of_int32<D0, D1, D2>(%arg0 : !mo.tensor<[D0, D1, D2], f32>) -> (!mo.tensor<[3], si32>) {
  %output = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si32>
  mo.output %output : !mo.tensor<[3], si32>
}
// CHECK{LITERAL}: 'shape_of_int32' returned tensor<3xsi32> [1, 2, 3]
