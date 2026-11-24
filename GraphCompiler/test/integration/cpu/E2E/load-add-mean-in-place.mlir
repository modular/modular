// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=true fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=true fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'load_add_mean':
// CHECK{LITERAL}: ([[2.0000, 2.0000, 2.0000]], dtype=f32, shape=[1,3])
mo.graph @load_add_mean(%buf: !mo.buffer<[2, 3], f32>, %add_input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 3], f32> {
  %ch0 = mo.chain.create()

  // Load tensor from buffer
  %loaded_tensor, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[2, 3], f32>, !mo.chain) -> (!mo.tensor<[2, 3], f32>, !mo.chain)

  // Add the loaded tensor with another tensor
  %added_tensor = mo.add(%loaded_tensor, %add_input) : !mo.tensor<[2, 3], f32>

  // Take mean along axis 0 (reduce the first dimension)
  %sum = mo.reduce.add(%added_tensor) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 3], f32>
  %len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[1, 3], f32>
  %mean_result = mo.div(%sum, %len_b) : !mo.tensor<[1, 3], f32>

  mo.output %mean_result : !mo.tensor<[1, 3], f32>
}
