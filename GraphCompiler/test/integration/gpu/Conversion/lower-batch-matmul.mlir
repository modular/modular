// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck --check-prefix=CHECK-MT %s

// CHECK-IR-LABEL: mef.callable @graph$exec
// CHECK-IR: %{{.*}} = "mgp.generic.execute"{{.*}}batch_matmul

// CHECK-MT-LABEL: --- Running 'graph':
// CHECK-MT: 'graph' returned tensor
// CHECK-MT-SAME: 2.0000, 2.0000
// CHECK-MT-NEXT: 2.0000, 2.0000
// CHECK-MT-NEXT: {{^$}}
// CHECK-MT-NEXT: 2.0000, 2.0000
// CHECK-MT-NEXT: 2.0000, 2.0000
// CHECK-MT-SAME: dtype=f32, shape=[2,2,2]


mo.graph @graph(%a : !mo.tensor<[2, 2, 2], f32>, %b : !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32>
{
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[2, 2, 2], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[2, 2, 2], f32> to <"gpu", 0>
  %c_gpu = mo.batch_matmul (%a_gpu, %b_gpu) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[2, 2, 2], f32, gpu:0>, !mo.tensor<[2, 2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2, 2], f32, gpu:0>
  %c, %ch3 = mo.transfer[%ch2] %c_gpu : !mo.tensor<[2, 2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[2, 2, 2], f32 >
}
