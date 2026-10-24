// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=3x3 --input-shapes=3x3 --input-shapes=3x3 %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'single_add':
// CHECK{LITERAL}: 'single_add' returned tensor([[2.0000, 2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000]], dtype=f32, shape=[3,3])
mo.graph @single_add<d0, d1, d2, d3, d4, d5>(%arg0: !mo.buffer<[d0, d1], f32>, %arg1: !mo.tensor<[d4, d5], f32>, %arg2: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[d0, d1], f32>
  %outTensor, %outChain = rmo.mo.mutable.load(%arg0, %0) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d2, d3], f32>, !mo.chain)
  %1 = rmo.mo.add(%outTensor, %arg1) : (!mo.tensor<[d2, d3], f32>, !mo.tensor<[d4, d5], f32>) -> !mo.tensor<[d4, d5], f32>
  %2 = rmo.mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d4, d5], f32>, !mo.chain) -> !mo.chain
  mo.output %arg0, %2 : !mo.buffer<[d0, d1], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'sequenced_add':
// CHECK{LITERAL}: 'sequenced_add' returned tensor([[3.0000, 3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000, 3.0000]], dtype=f32, shape=[3,3])
mo.graph @sequenced_add<d0, d1, d2, d3, d4, d5>(%arg0: !mo.buffer<[d0, d1], f32>, %arg1: !mo.tensor<[d4, d5], f32>, %arg2: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[d0, d1], f32>

  %outTensor, %outChain = rmo.mo.mutable.load(%arg0, %0) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d2, d3], f32>, !mo.chain)
  %1 = rmo.mo.add(%outTensor, %arg1) : (!mo.tensor<[d2, d3], f32>, !mo.tensor<[d4, d5], f32>) -> !mo.tensor<[d4, d5], f32>
  %2 = rmo.mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d4, d5], f32>, !mo.chain) -> !mo.chain

  %outTensor2, %ch3 = rmo.mo.mutable.load(%arg0, %2) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d2, d3], f32>, !mo.chain)
  %res2 = rmo.mo.add(%outTensor2, %arg1) : (!mo.tensor<[d2, d3], f32>, !mo.tensor<[d4, d5], f32>) -> !mo.tensor<[d4, d5], f32>
  %ch4 = rmo.mo.mutable.store(%arg0, %res2, %ch3) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d4, d5], f32>, !mo.chain) -> !mo.chain

  mo.output %arg0, %ch4 : !mo.buffer<[d0, d1], f32>, !mo.chain
}
