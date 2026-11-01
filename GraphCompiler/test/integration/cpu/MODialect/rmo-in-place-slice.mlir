// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=5x5 --input-shapes=3x3 --input-shapes=2 %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'strided_slice':
// CHECK{LITERAL}: 'strided_slice' returned tensor([[1.0000, -1.0000, 1.0000, -1.0000, 1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
// CHECK{LITERAL}: [1.0000, -1.0000, 1.0000, -1.0000, 1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
// CHECK{LITERAL}: [1.0000, -1.0000, 1.0000, -1.0000, 1.0000]], dtype=f32, shape=[5,5])
mo.graph @strided_slice<d0, d1>(%arg0: !mo.buffer<[5, 5], f32>, %arg1: !mo.tensor<[d0, d1], f32>, %arg2: !mo.tensor<[2], si64>) -> (!mo.buffer<[5, 5], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[5, 5], f32>
  %1 = mo.add(%arg2, %arg2) : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.constant <#M.dense_array<5, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[5, 5], f32>, !mo.chain) -> (!mo.tensor<[5, 5], f32>, !mo.chain)
  %4 = mo.negative(%outTensor) : !mo.tensor<[5, 5], f32>
  %5 = mo.mutable.store(%arg0, %4, %outChain) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[5, 5], f32>, !mo.chain) -> !mo.chain
  %6 = rmo.mo.mutable.store.slice(%arg0, %arg1, %2, %3, %1, %5) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[d0, d1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %6 : !mo.buffer<[5, 5], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'strided_slice_constant':
// CHECK{LITERAL}: 'strided_slice_constant' returned tensor([[1.0000, -1.0000, 1.0000, -1.0000, 1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
// CHECK{LITERAL}: [1.0000, -1.0000, 1.0000, -1.0000, 1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
// CHECK{LITERAL}: [1.0000, -1.0000, 1.0000, -1.0000, 1.0000]], dtype=f32, shape=[5,5])
mo.graph @strided_slice_constant<d0, d1>(%arg0: !mo.buffer<[5, 5], f32>, %arg1: !mo.tensor<[d0, d1], f32>) -> (!mo.buffer<[5, 5], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[5, 5], f32>
  %2 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.constant <#M.dense_array<5, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[5, 5], f32>, !mo.chain) -> (!mo.tensor<[5, 5], f32>, !mo.chain)
  %4 = mo.negative(%outTensor) : !mo.tensor<[5, 5], f32>
  %5 = mo.mutable.store(%arg0, %4, %outChain) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[5, 5], f32>, !mo.chain) -> !mo.chain
  %6 = rmo.mo.mutable.store.slice(%arg0, %arg1, %2, %3, %step, %5) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[d0, d1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %6 : !mo.buffer<[5, 5], f32>, !mo.chain
}
