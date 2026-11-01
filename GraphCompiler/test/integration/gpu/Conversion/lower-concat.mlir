// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %s | FileCheck %s

// CHECK-LABEL: --- Running 'concat':
// CHECK: 'concat' returned tensor<2x1x4xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]

mo.graph @concat(%arg0: !mo.tensor<[1, 1, 4], f32>) -> !mo.tensor<[2, 1, 4], f32> {
  %ch0 = mo.chain.create()
  %1, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[1, 1, 4], f32> to <"gpu", 0>
  %in1 = mo.add (%1, %1) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[1, 1, 4], f32, gpu:0>
  %in2 = mo.add (%in1, %1) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[1, 1, 4], f32, gpu:0>
  %5 = mogg.concat (%in2, %in1) : (!mo.tensor<[1, 1, 4], f32, gpu:0>, !mo.tensor<[1, 1, 4], f32, gpu:0>) -> !mo.tensor<[2, 1, 4], f32, gpu:0>
  %6, %ch2 = mo.transfer[%ch1] %5 : !mo.tensor<[2, 1, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %6 : !mo.tensor<[2, 1, 4], f32>
}
