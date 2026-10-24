// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'scalar_constant_sub':
// CHECK: 'scalar_constant_sub' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @scalar_constant_sub(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %s = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %ch0 = mo.chain.create()
  %s_gpu, %ch1 = mo.transfer[%ch0] %s : !mo.tensor<[], f32> to <"gpu", 0>
  %x_gpu, %ch2 = mo.transfer[%ch1] %x : !mo.tensor<[2, 2], f32> to <"gpu", 0>
  %b = mo.static.broadcast_to(%s_gpu) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[2, 2], f32, gpu:0>
  %r_gpu = mo.sub(%b, %x_gpu) : !mo.tensor<[2, 2], f32, gpu:0>
  %r, %ch3 = mo.transfer[%ch2] %r_gpu : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'broadcast_scalar_constant':
// CHECK: 'broadcast_scalar_constant' returned tensor<2x3xf32> [7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00, 7.500000e+00]
mo.graph @broadcast_scalar_constant() -> (!mo.tensor<[2, 3], f32>) {
  %s = mo.constant <#M.dense_array<7.500000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %ch0 = mo.chain.create()
  %s_gpu, %ch1 = mo.transfer[%ch0] %s : !mo.tensor<[], f32> to <"gpu", 0>
  %b = mo.static.broadcast_to(%s_gpu) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[2, 3], f32, gpu:0>
  %r, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[2, 3], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[2, 3], f32>
}
