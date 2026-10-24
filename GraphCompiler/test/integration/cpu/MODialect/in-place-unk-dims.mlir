// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --input-shapes=4x4 --input-shapes=2x2 %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'unk_dims':
// CHECK{LITERAL}: 'unk_dims' returned tensor([[1.0000, 2.0000, 1.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000, 2.0000],
// CHECK{LITERAL}: [1.0000, 2.0000, 1.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4,4])
mo.graph @unk_dims<D0, D1, D2, D3, D4, D5>(%arg0: !mo.buffer<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>, %slice: !mo.tensor<[D4, D5], f32>) -> (!mo.buffer<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>, !mo.chain) {
  %0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %1 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %2 = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %3 = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %4 = mo.constant <#M.dense_array<2> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %5 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  %c0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %c1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %c2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %6 = mo.chain.create()
  %outTensor, %outChain = rmo.mo.mutable.load(%arg0, %6) : (!mo.buffer<[D0, D1], f32>, !mo.chain) -> (!mo.tensor<[D0, D1], f32>, !mo.chain)
  %casted = rmo.mo.cast(%0) : (!mo.tensor<[], si64>) -> !mo.tensor<[], f32>
  %arg1shape = rmo.mo.shape_of(%arg1) : (!mo.tensor<[D2, D3], f32>) -> !mo.tensor<[2], si64>
  %bcasted = rmo.mo.broadcast_to(%casted, %arg1shape) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32>
  %7 = rmo.mo.mul(%arg1, %bcasted) : (!mo.tensor<[D2, D3], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D2, D3], f32>
  %8 = rmo.mo.add(%outTensor, %7) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  %9 = rmo.rebind_tensor_shape(%outTensor) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  %10 = rmo.mo.shape_of(%9) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64>
  %11 = rmo.mo.slice(%10, %1, %2, %2) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %13 = rmo.mo.reshape(%11, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[], si64>
  %14 = rmo.rebind_tensor_shape(%outTensor) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  %15 = rmo.mo.shape_of(%14) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64>
  %16 = rmo.mo.slice(%15, %2, %4, %2) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %17 = rmo.mo.reshape(%16, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[], si64>
  %18 = rmo.mo.reshape(%13, %2) : (!mo.tensor<[], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %19 = rmo.mo.reshape(%17, %2) : (!mo.tensor<[], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %20 = mo.concat<() -> D6>[0] (%18, %19) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[D6], si64>
  %21 = rmo.mo.shape_of(%8) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[2], si64>
  %22 = mo.concat[0] (%1, %1) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %23 = rmo.mo.pad.constant(%21, %22, %5) : (!mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2], si64>
  %24 = rmo.rebind_tensor_shape(%20) : (!mo.tensor<[D6], si64>) -> !mo.tensor<[2], si64>
  %25 = rmo.mo.max(%23, %24) : (!mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si64>
  %26 = rmo.mo.broadcast_to(%8, %25) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32>
  %27 = rmo.mo.mutable.store(%arg0, %26, %outChain) : (!mo.buffer<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>, !mo.chain) -> !mo.chain
  %res, %28 = rmo.mo.mutable.load(%arg0, %27) : (!mo.buffer<[D0, D1], f32>, !mo.chain) -> (!mo.tensor<[D0, D1], f32>, !mo.chain)
  %29 = rmo.mo.mutable.store.slice(%arg0, %slice, %c0, %c1, %c2, %28) : (!mo.buffer<[D0, D1], f32>, !mo.tensor<[D4, D5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain
  // COM: Output %res as well to catch any bugs with the interaction b/n parameterization of mo.output operands and mo.buffer values
  mo.output %arg0, %res, %29 : !mo.buffer<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>, !mo.chain
}
