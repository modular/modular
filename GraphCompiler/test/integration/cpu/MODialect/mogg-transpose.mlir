// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute %t.mlir -model-inputs=ones | FileCheck %s

mo.graph @transpose() -> !mo.tensor<[5, 4], si32>  {
  %0 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20> : tensor<20xsi32>> : !mo.tensor<[4, 5], si32>
  %perm = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>

  %trans = mo.transpose(%0, %perm) : (!mo.tensor<[4, 5], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 4], si32>
  mo.output %trans : !mo.tensor<[5, 4], si32>
}

// CHECK-LABEL: --- Running 'transpose':
// CHECK-NEXT{LITERAL}: 'transpose' returned tensor([[1, 6, 11, 16],
// CHECK-NEXT{LITERAL}: [2, 7, 12, 17],
// CHECK-NEXT{LITERAL}: [3, 8, 13, 18],
// CHECK-NEXT{LITERAL}: [4, 9, 14, 19],
// CHECK-NEXT{LITERAL}: [5, 10, 15, 20]], dtype=si32, shape=[5,4])


mo.graph @transpose_fused(%arg0 : !mo.tensor<[5, 4], si32>) -> !mo.tensor<[5, 4], si32>  {
  %0 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20> : tensor<20xsi32>> : !mo.tensor<[4, 5], si32>
  %perm = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>

  %trans = mo.transpose(%0, %perm) : (!mo.tensor<[4, 5], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 4], si32>
  %add = mo.add(%arg0, %trans) : !mo.tensor<[5, 4], si32>
  mo.output %add : !mo.tensor<[5, 4], si32>
}

// CHECK-LABEL: --- Running 'transpose_fused':
// CHECK-NEXT{LITERAL}: 'transpose_fused' returned tensor([[2, 7, 12, 17],
// CHECK-NEXT{LITERAL}: [3, 8, 13, 18],
// CHECK-NEXT{LITERAL}: [4, 9, 14, 19],
// CHECK-NEXT{LITERAL}: [5, 10, 15, 20],
// CHECK-NEXT{LITERAL}: [6, 11, 16, 21]], dtype=si32, shape=[5,4])

mo.graph @transpose_and_bcast_fused(%arg0 : !mo.tensor<[], si32>) -> !mo.tensor<[5, 4], si32>  {
  %0 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20> : tensor<20xsi32>> : !mo.tensor<[4, 5], si32>
  %perm = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %trans = mo.transpose(%0, %perm) : (!mo.tensor<[4, 5], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 4], si32>
  %bcast = mo.static.broadcast_to(%arg0) : !mo.tensor<[], si32> -> !mo.tensor<[5, 4], si32>
  %add = mo.add(%bcast, %trans) : !mo.tensor<[5, 4], si32>
  mo.output %add : !mo.tensor<[5, 4], si32>
}

// CHECK-LABEL: --- Running 'transpose_and_bcast_fused':
// CHECK-NEXT{LITERAL}: 'transpose_and_bcast_fused' returned tensor([[2, 7, 12, 17],
// CHECK-NEXT{LITERAL}: [3, 8, 13, 18],
// CHECK-NEXT{LITERAL}: [4, 9, 14, 19],
// CHECK-NEXT{LITERAL}: [5, 10, 15, 20],
// CHECK-NEXT{LITERAL}: [6, 11, 16, 21]], dtype=si32, shape=[5,4])
