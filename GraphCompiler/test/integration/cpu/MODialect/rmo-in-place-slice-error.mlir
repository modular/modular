// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=5x5 --input-shapes=5x5 --input-shapes=2 %t.mlir > %t.output || true
// RUN: cat %t.output | FileCheck %s

// COM: with the given start/stop/step of [0,0]/[5,5]/[2,2] we expect a slice shape of 3x3 rather than the specified 5x5 above

// CHECK-LABEL: --- Running 'bad_shape':
// CHECK: <error:
// CHECK: Shape mismatch for mo.mutable.store.slice: expected 'slice' operand to have shape: (3, 3) but got: (5, 5)
mo.graph @bad_shape<d0, d1>(%arg0: !mo.buffer<[5, 5], f32>, %arg1: !mo.tensor<[d0, d1], f32>, %arg2: !mo.tensor<[2], si64>) -> (!mo.buffer<[5, 5], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[5, 5], f32>
  %1 = mo.add(%arg2, %arg2) : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.constant <#M.dense_array<5, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[5, 5], f32>, !mo.chain) -> (!mo.tensor<[5, 5], f32>, !mo.chain)
  %4 = mo.negative(%outTensor) : !mo.tensor<[5, 5], f32>
  %5 = mo.mutable.store(%arg0, %4, %outChain) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[5, 5], f32>, !mo.chain) -> !mo.chain
  %6 = rmo.mo.mutable.store.slice(%arg0, %arg1, %2, %3, %1, %0) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[d0, d1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %6 : !mo.buffer<[5, 5], f32>, !mo.chain
}
