// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=true fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=true fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'load_matmul_add_store':
// CHECK{LITERAL}: ([[5.0000, 5.0000, 5.0000, 5.0000],
// CHECK{LITERAL}: [5.0000, 5.0000, 5.0000, 5.0000],
// CHECK{LITERAL}: [5.0000, 5.0000, 5.0000, 5.0000],
// CHECK{LITERAL}: [5.0000, 5.0000, 5.0000, 5.0000]], dtype=f32, shape=[4,4])
mo.graph @load_matmul_add_store(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> (!mo.buffer<[4, 4], f32>, !mo.chain) {
  %ch0 = mo.chain.create()

  %loaded_tensor, %ch1 = mo.mutable.load(%buf0, %ch0) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %1 = mo.matmul(%loaded_tensor, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %2 = mo.add(%1, %arg2) : !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%buf0, %2, %ch1) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  mo.output %buf0, %ch2 : !mo.buffer<[4, 4], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'load_matmul_store_twice':
// CHECK{LITERAL}: ([[16.0000, 16.0000, 16.0000, 16.0000],
// CHECK{LITERAL}: [16.0000, 16.0000, 16.0000, 16.0000],
// CHECK{LITERAL}: [16.0000, 16.0000, 16.0000, 16.0000],
// CHECK{LITERAL}: [16.0000, 16.0000, 16.0000, 16.0000]], dtype=f32, shape=[4,4])
mo.graph @load_matmul_store_twice(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> (!mo.buffer<[4, 4], f32>, !mo.chain) {
  %ch0 = mo.chain.create()

  %loaded_tensor1, %ch1 = mo.mutable.load(%buf0, %ch0) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %1 = mo.matmul(%loaded_tensor1, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%buf0, %1, %ch1) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  %loaded_tensor2, %ch3 = mo.mutable.load(%buf0, %ch2) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %2 = mo.matmul(%loaded_tensor2, %arg2) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch4 = mo.mutable.store(%buf0, %2, %ch3) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  mo.output %buf0, %ch4 : !mo.buffer<[4, 4], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'matmul_cast_add_store':
// CHECK{LITERAL}: ([[5, 5, 5, 5],
// CHECK{LITERAL}: [5, 5, 5, 5],
// CHECK{LITERAL}: [5, 5, 5, 5],
// CHECK{LITERAL}: [5, 5, 5, 5]], dtype=si32, shape=[4,4])
mo.graph @matmul_cast_add_store(%buf0: !mo.buffer<[4, 4], si32>, %arg1: !mo.tensor<[4, 4], f16>, %arg2: !mo.tensor<[4, 4], si32>) -> (!mo.buffer<[4, 4], si32>, !mo.chain) {
  %ch0 = mo.chain.create()

  %1 = mo.matmul(%arg1, %arg1) : (!mo.tensor<[4, 4], f16>, !mo.tensor<[4, 4], f16>) -> !mo.tensor<[4, 4], f16>
  %2 = mo.cast(%1) : (!mo.tensor<[4, 4], f16>) -> !mo.tensor<[4, 4], si32>
  %3 = mo.add(%2, %arg2) : !mo.tensor<[4, 4], si32>
  %ch2 = mo.mutable.store(%buf0, %3, %ch0) : (!mo.buffer<[4, 4], si32>, !mo.tensor<[4, 4], si32>, !mo.chain) -> (!mo.chain)

  mo.output %buf0, %ch2 : !mo.buffer<[4, 4], si32>, !mo.chain
}
