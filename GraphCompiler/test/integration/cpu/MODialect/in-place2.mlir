// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=true fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=false fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=true fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=false fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!c = !mo.chain

// CHECK-LABEL: --- Running 'basic_slice_store':
// CHECK{LITERAL}: 'basic_slice_store' returned tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 0.0000, 0.0000, 0.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 0.0000, 0.0000, 0.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 0.0000, 0.0000, 0.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[5,5])
mo.graph @basic_slice_store(%arg0: !mo.buffer<[5, 5], f32>, %arg1: !mo.tensor<[5, 5], f32>) -> (!mo.buffer<[5, 5], f32>, !c) {
  %0 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %4 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32> -> !mo.tensor<[3, 3], f32>
  %5 = mo.chain.create(%arg0) : !mo.buffer<[5, 5], f32>
  %6 = mo.mutable.store.slice(%arg0, %4, %0, %1, %2, %5) : (!mo.buffer<[5, 5], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %6 : !mo.buffer<[5, 5], f32>, !c
}

// CHECK-LABEL: --- Running 'strided_compute':
// CHECK{LITERAL}: 'strided_compute' returned tensor([[-2.0000, 1.0000, -2.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000, 1.0000, 1.0000],
// CHECK{LITERAL}: [-2.0000, 1.0000, -2.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[4,4])
mo.graph @strided_compute(%arg0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[2, 2], f32>) -> (!mo.buffer<[4, 4], f32>, !c) {
  %0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.chain.create(%arg0) : !mo.buffer<[4, 4], f32>
  %outTensor, %outChain = mo.mutable.load(%arg0, %3) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %4 = mo.slice(%outTensor, %0, %1, %2) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.add(%4, %arg1) : !mo.tensor<[2, 2], f32>
  %6 = mo.negative(%5) : !mo.tensor<[2, 2], f32>
  %7 = mo.mutable.store.slice(%arg0, %6, %0, %1, %2, %outChain) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  mo.output %arg0, %7: !mo.buffer<[4, 4], f32>, !c
}

// CHECK-LABEL: --- Running 'multiple_loads_to_same_op':
// CHECK{LITERAL}: 'multiple_loads_to_same_op' returned tensor([[2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
// CHECK{LITERAL}: , tensor([[1.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000]], dtype=f32, shape=[2,2])
mo.graph @multiple_loads_to_same_op(%blhs: !bt, %brhs: !bt) -> (!bt, !bt, !c)
{
  %ch0 = mo.chain.create()

  %lhs, %ch1 = mo.mutable.load(%blhs, %ch0) : (!bt, !c) -> (!tt, !c)
  %rhs, %ch1p = mo.mutable.load(%brhs, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%lhs, %rhs) : !tt
  %ch2 = mo.mutable.store(%blhs, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  mo.output %blhs, %brhs, %ch2 : !bt, !bt, !c
}

// CHECK-LABEL: --- Running 'multiple_loads':
// CHECK{LITERAL}: 'multiple_loads' returned tensor([[-1.0000, -1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000]], dtype=f32, shape=[2,2])
// CHECK{LITERAL}: , tensor([[0.5403, 0.5403],
// CHECK{LITERAL}: [0.5403, 0.5403]], dtype=f32, shape=[2,2])
// CHECK{LITERAL}: , tensor([[2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
mo.graph @multiple_loads(%buf: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !c)
{
  %0 = mo.chain.create()
  %tensor0, %ch0 = mo.mutable.load(%buf, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %tensor1, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %tensor2, %ch2 = mo.mutable.load(%buf, %ch1) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %tensor3, %ch3 = mo.mutable.load(%buf, %ch2) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)

  %res1 = mo.negative(%tensor0) : !mo.tensor<[2, 2], f32>
  %res2 = mo.cos(%tensor1) : !mo.tensor<[2, 2], f32>
  %res3 = mo.add(%tensor2, %tensor3) : !mo.tensor<[2, 2], f32>

  %ch4 = mo.mutable.store(%buf, %res3, %ch3) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain

  mo.output %res1, %res2, %buf, %ch4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !c
}
