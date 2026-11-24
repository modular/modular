// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=4 --input-data=0::ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @to_tensor_from_tensor(%arg: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si32> {
  %mask = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %val = mo.add(%arg, %mask) : !mo.tensor<[4], si64>

  %sh = mo.shape.from_tensor(%val) : (!mo.tensor<[4], si64>) -> !mosh.ape
  %tensor = mo.shape.to_tensor(%sh) -> !mo.tensor<[4], si32>

  mo.output %tensor : !mo.tensor<[4], si32>
}
// CHECK{LITERAL}: 'to_tensor_from_tensor' returned tensor<4xsi32> [1, 2, 3, 4]
