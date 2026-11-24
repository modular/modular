// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x128x768 --input-shapes=1 --model-inputs=ones %t.mlir | FileCheck %s

// Check the intersection of MOGG and shape lowers correctly.
mo.graph @broadcastTest<D0, D1>(%arg0: !mo.tensor<[D0, D1, 768], f32>, %arg1: !mo.tensor<[1], f32>) -> (!mo.tensor<[D0, D1, 768], f32>) {
  %shape1 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, 768], f32>) -> !mo.tensor<[3], si64>
  %shape2 = mo.shape_of(%arg1) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], si64>
  %shape = mo.broadcast_shape(%shape1, %shape2) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si64>
  %0 = mo.broadcast_to(%arg0, %shape) : (!mo.tensor<[D0, D1, 768], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D0, D1, 768], f32>
  %1 = mo.broadcast_to(%arg1, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D0, D1, 768], f32>
  %3 = mo.add(%0, %1) : !mo.tensor<[D0, D1, 768], f32>
  %4 = mo.add(%0, %3) : !mo.tensor<[D0, D1, 768], f32>
  %5 = mo.mul(%3, %4) : !mo.tensor<[D0, D1, 768], f32>
  %6 = mo.sqrt(%5) : !mo.tensor<[D0, D1, 768], f32>
  mo.output %6 : !mo.tensor<[D0, D1, 768], f32>
}

// Above is sqrt(((1 + 1) + 1) * (1 + 1)) == sqrt(6) == 2.4495


// CHECK{LITERAL}: 'broadcastTest' returned tensor([[[2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],

// CHECK{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495]],

// CHECK{LITERAL}: [[2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],

// CHECK{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495],
// CHECK-NEXT{LITERAL}: [2.4495, 2.4495, 2.4495, ..., 2.4495, 2.4495, 2.4495]]], dtype=f32, shape=[2,128,768])
