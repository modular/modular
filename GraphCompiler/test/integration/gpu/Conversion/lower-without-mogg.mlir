// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck %s

// This exercises the MO CUDA lowering support but without any dependence on MO ops which
// must go via MOGG.

// CHECK-LABEL: --- Running 'print_string':
// CHECK-NEXT: [gpu_test1] This is the first string
// CHECK-NEXT: [gpu_test2] This is the second string
// CHECK-NEXT: 'print_string' returned tensor({{\[}}[1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[4])

mo.graph @print_string(%a : !mo.tensor<[4], f32, gpu:0>) -> (!mo.tensor<[4], f32, gpu:0>, !mo.chain)
{
  %b = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0> : tensor<4xf32>> : !mo.tensor<[4], f32, gpu:0>
  %ch0 = mo.chain.create()
  %ch1 = mo.debug.print(%ch0) [value = "This is the first string", label = "gpu_test1"] {device = #M.device_ref<"gpu", 0>}
  %ch2 = mo.debug.print(%ch1) [value = "This is the second string", label = "gpu_test2"] {device = #M.device_ref<"gpu", 0>}
  mo.output %a, %ch2 : !mo.tensor<[4], f32, gpu:0>, !mo.chain
}


// CHECK-LABEL: --- Running 'print':
// CHECK-NEXT: Tensor(4xf32, TensorBufferRef(0x
// CHECK-NEXT: Tensor(4xf32, TensorBufferRef(0x
// CHECK-NEXT: 'print' returned tensor({{\[}}[1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[4])

mo.graph @print(%a : !mo.tensor<[4], f32, gpu:0>) -> (!mo.tensor<[4], f32, gpu:0>, !mo.chain)
{
  %b = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0> : tensor<4xf32>> : !mo.tensor<[4], f32, gpu:0>
  %ch0 = mo.chain.create()
  %ch1 = mo.debug.tensor.print(%ch0, %a) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  %ch2 = mo.debug.tensor.print(%ch1, %b) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  mo.output %a, %ch2 : !mo.tensor<[4], f32, gpu:0>, !mo.chain
}
