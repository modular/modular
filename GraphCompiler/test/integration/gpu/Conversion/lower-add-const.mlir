// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck %s

// CHECK-LABEL: --- Running 'graph':
// CHECK: 'graph' returned tensor({{\[}}[1.0000, 2.0000, 3.0000, 4.0000]], dtype=f32, shape=[4])

mo.graph @graph(%a : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %b = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %c_gpu = mo.add (%a_gpu, %b_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  %c, %ch3 = mo.transfer[%ch2] %c_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[4], f32 >
}
