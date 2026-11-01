
// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'matmul_cast':
// CHECK: 'matmul_cast' returned tensor<4x4xsi32> [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

mo.graph @matmul_cast(%a : !mo.tensor<[4, 4], f32>, %b: !mo.tensor<[4, 4], f32>, %c: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %1 = mo.add(%0, %c) : !mo.tensor<[4, 4], f32>
  %2 = mo.cast(%1) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32>
  mo.output %2 : !mo.tensor<[4, 4], si32>
}
