// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck --check-prefix=CHECK-MT %s

// CHECK-IR-LABEL: mef.callable @graph$exec
// CHECK-IR: %{{.*}} = "mgp.generic.execute"{{.*}}scatter_nd

// CHECK-MT-LABEL: --- Running 'graph':
// CHECK-MT: 'graph' returned tensor({{\[}}[1.0000, 1.0000],
// CHECK-MT-NEXT: [1.0000, 1.0000],
// CHECK-MT-NEXT: [1.0000, 1.0000],
// CHECK-MT-NEXT: [1.0000, 1.0000]], dtype=f32, shape=[4,2])

mo.graph @graph(%a: !mo.tensor<[4, 2], f32>, %b: !mo.tensor<[1, 3, 2], f32>, %c: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
{
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4, 2], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[1, 3, 2], f32> to <"gpu", 0>
  %c_gpu, %ch3 = mo.transfer[%ch2] %c : !mo.tensor<[1, 3, 1], si64> to <"gpu", 0>
  %d_gpu = mo.scatter_nd(%a_gpu, %b_gpu, %c_gpu) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[1, 3, 2], f32, gpu:0>, !mo.tensor<[1, 3, 1], si64, gpu:0>) -> !mo.tensor<[4, 2], f32, gpu:0>
  %d, %ch4 = mo.transfer[%ch3] %d_gpu : !mo.tensor<[4, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %d : !mo.tensor<[4, 2], f32>
}
