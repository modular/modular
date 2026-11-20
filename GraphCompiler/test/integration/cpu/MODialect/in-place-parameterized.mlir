// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'parameterized':
// CHECK{LITERAL}: 'parameterized' returned tensor([[-2.0000, 1.0000, -2.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000, 1.0000, 1.0000],
// CHECK{LITERAL}: [-2.0000, 1.0000, -2.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[4,4])
mo.graph @parameterized<d0, d1, d2, d3>(%arg0: !mo.buffer<[d0, d1], f32>, %arg1: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.chain) {
  %0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.chain.create(%arg0) : !mo.buffer<[d0, d1], f32>
  %outTensor, %outChain = mo.mutable.load(%arg0, %3) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
  %4 = mo.slice(%outTensor, %0, %1, %2) : (!mo.tensor<[d0, d1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[d2, d3], f32>
  %5 = mo.add(%4, %arg1) : !mo.tensor<[d2, d3], f32>
  %6 = mo.negative(%5) : !mo.tensor<[d2, d3], f32>
  %7 = mo.mutable.store.slice(%arg0, %6, %0, %1, %2, %outChain) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %7 : !mo.buffer<[d0, d1], f32>, !mo.chain
}
