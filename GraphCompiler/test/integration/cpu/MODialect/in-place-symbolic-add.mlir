// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=3x3 --input-shapes=3x3 %t.mlir | FileCheck %s

// COM: The added tensor declares its own symbolic dims; the add's lowering
// COM: asserts at runtime that they match the buffer's.

// CHECK-LABEL: --- Running 'single_add':
// CHECK{LITERAL}: 'single_add' returned tensor([[2.0000, 2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000]], dtype=f32, shape=[3,3])
mo.graph @single_add<d0, d1, d2, d3>(%arg0: !mo.buffer<[d0, d1], f32>, %arg1: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[d0, d1], f32>
  %loaded, %ch1 = mo.mutable.load(%arg0, %0) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
  %sum = rmo.mo.add(%loaded, %arg1) : (!mo.tensor<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
  %ch2 = mo.mutable.store(%arg0, %sum, %ch1) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
  mo.output %arg0, %ch2 : !mo.buffer<[d0, d1], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'sequenced_add':
// CHECK{LITERAL}: 'sequenced_add' returned tensor([[3.0000, 3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000, 3.0000]], dtype=f32, shape=[3,3])
mo.graph @sequenced_add<d0, d1, d2, d3>(%arg0: !mo.buffer<[d0, d1], f32>, %arg1: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.chain) {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[d0, d1], f32>

  %loaded, %ch1 = mo.mutable.load(%arg0, %0) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
  %sum = rmo.mo.add(%loaded, %arg1) : (!mo.tensor<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
  %ch2 = mo.mutable.store(%arg0, %sum, %ch1) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain

  %loaded2, %ch3 = mo.mutable.load(%arg0, %ch2) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
  %sum2 = rmo.mo.add(%loaded2, %arg1) : (!mo.tensor<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
  %ch4 = mo.mutable.store(%arg0, %sum2, %ch3) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain

  mo.output %arg0, %ch4 : !mo.buffer<[d0, d1], f32>, !mo.chain
}
