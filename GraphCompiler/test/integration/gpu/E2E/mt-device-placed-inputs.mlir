// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --num-runs=2 --result-output-style=compact %s | FileCheck %s --check-prefix=RERUN

// mt generates inputs on the host and stages any device-placed input onto its
// declared device before execution. The gpu-placed input %a is consumed by a
// device kernel, so the checked values prove the staged copy delivered the
// generated data (ones + ones = twos), unlike a metadata-only check that any
// pointer would satisfy. The second RUN re-stages the inputs on every run.
// RERUN-COUNT-2: 'main' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])
mo.graph @main(%a : !mo.tensor<[4], f32, gpu:0>, %b : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %b_gpu, %ch1 = mo.transfer[%ch0] %b : !mo.tensor<[4], f32> to <"gpu", 0>
  %c_gpu = mo.add (%a, %b_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[4], f32, gpu:0>
  %c, %ch2 = mo.transfer[%ch1] %c_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[4], f32>
}
