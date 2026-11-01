// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck %s

// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'graph':
// CHECK: 'graph' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])

mo.graph @graph(%a : !mo.tensor<[4], f32>, %b : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %c_gpu = mo.add (%a_gpu, %b_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  %c, %ch3 = mo.transfer[%ch2] %c_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[4], f32>
}
