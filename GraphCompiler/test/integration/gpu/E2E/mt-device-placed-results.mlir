// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck %s

// mt reads results on the host: a device-placed result is copied back to a
// host buffer after execution and printed from there. Combined with the
// gpu-placed input, the model's data flow is device-resident end to end;
// the checked values prove both staging copies moved real data
// (ones + ones = twos).

// CHECK-LABEL: --- Running 'device_result':
// CHECK: 'device_result' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])
mo.graph @device_result(%a : !mo.tensor<[4], f32, gpu:0>, %b : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32, gpu:0> {
  %ch0 = mo.chain.create()
  %b_gpu, %ch1 = mo.transfer[%ch0] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %c_gpu = mo.add (%a, %b_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  mo.output %c_gpu : !mo.tensor<[4], f32, gpu:0>
}
