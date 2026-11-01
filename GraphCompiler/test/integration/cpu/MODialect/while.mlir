// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK: --- Running 'while_op':
// CHECK-NEXT: 'while_op' returned tensor<2xf32> [4.000000e+00, 4.000000e+00]
mo.graph @while_op() -> (!mo.tensor<[2], f32>) {
   %x = mo.constant <#M.dense_array<1.0, 1.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
   %y = mo.constant <#M.dense_array<4.0> : tensor<1xf32>> : !mo.tensor<[], f32>
   %res = mo.while (%x as %xi: !mo.tensor<[2], f32>) {
     %sum0 = mo.reduce.add(%xi) : (!mo.tensor<[2], f32>) -> !mo.tensor<[1], f32>
     %len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
     %mean0 = mo.div(%sum0, %len) : !mo.tensor<[1], f32>
     %mean1 = mo.static.reshape(%mean0) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
     %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.while.condition(%cond) %xi : !mo.tensor<[2], f32>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[2], f32>
     mo.yield %new_x : !mo.tensor<[2], f32>
   }
   mo.output %res : !mo.tensor<[2], f32>
}
