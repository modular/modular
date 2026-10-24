// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute %t.mlir --input-shapes=2x2x3 -model-inputs=ones | FileCheck %s

mo.graph @many_bcasts<N, M, L>(%in : !mo.tensor<[N, M, L], f32>) -> !mo.tensor<[N, M, L], f32> {
  %1 = mo.constant <#M.dense_array<2.000000e+00, 0.000000e+00> : tensor<2xf32>> : !mo.tensor<[2, 1, 1], f32>
  %2 = mo.constant <#M.dense_array<5.000000e-01, 2.500000e-01> : tensor<2xf32>> : !mo.tensor<[2, 1], f32>
  %3 = mo.constant <#M.dense_array<1.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<3xf32>> : !mo.tensor<[3], f32>

  %bcast = mo.static.broadcast_to(%1) : !mo.tensor<[2, 1, 1], f32> -> !mo.tensor<[N, M, L], f32>
  %bcast2 = mo.static.broadcast_to(%2) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[N, M, L], f32>
  %bcast3 = mo.static.broadcast_to(%3) : !mo.tensor<[3], f32> -> !mo.tensor<[N, M, L], f32>

  %out = mo.mul(%bcast, %in) : !mo.tensor<[N, M, L], f32>
  %out2 = mo.add(%bcast2, %out) : !mo.tensor<[N, M, L], f32>
  %out3 = mo.sub(%out2, %bcast3) : !mo.tensor<[N, M, L], f32>

  mo.output %out3 : !mo.tensor<[N, M, L], f32>
}

// tensor all ones
// tensor[0][:][:] * 2
// tensor[1][:][:] * 0

// tensor[:][0][:] + 0.5
// tensor[:][1][:] + 0.25

// tensor[:][:][0] - 1
// tensor[:][:][1] - 7
// tensor[:][:][2] - 8

// CHECK-IR: mgp.generic.execute
// CHECK{LITERAL}: 'many_bcasts' returned tensor([[[1.5000, -4.5000, -5.5000],
// CHECK-NEXT{LITERAL}: [1.2500, -4.7500, -5.7500]],
// CHECK{LITERAL}: [[-0.5000, -6.5000, -7.5000],
// CHECK-NEXT{LITERAL}: [-0.7500, -6.7500, -7.7500]]], dtype=f32, shape=[2,2,3])
