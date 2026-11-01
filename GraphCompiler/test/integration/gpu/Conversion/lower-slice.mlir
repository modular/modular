// FIXME: GEX-2153
// UNSUPPORTED: AMD-GPU

// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s -o %t.mlir

// RUN: cat %t.mlir | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %t.mlir | FileCheck --check-prefix=CHECK-MT %s

// CHECK-IR-LABEL: mef.callable @graph$exec
// CHECK-IR: %{{.*}} = "mgp.generic.execute"{{.*}}slice

// CHECK-MT-LABEL: --- Running 'graph':
// CHECK-MT: 'graph' returned tensor({{\[}}{{\[}}1.0000, 1.0000, 1.0000],
// CHECK-MT-NEXT: [1.0000, 1.0000, 1.0000],
// CHECK-MT-NEXT: [1.0000, 1.0000, 1.0000]], dtype=f32, shape=[3,3])

mo.graph @graph<() -> D0, D1>(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[D0, D1], f32> {
  %start = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %ch0 = mo.chain.create()
  %input_gpu, %ch1 = mo.transfer[%ch0] %input : !mo.tensor<[10, 10], f32> to <"gpu", 0>
  %res_gpu = mo.slice<() -> D2, D3>(%input_gpu, %start, %stop, %step) { device = #M.device_ref<"gpu", 0> } : (
    !mo.tensor<[10, 10], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D2, D3], f32, gpu:0>
  %res, %ch2 = mo.transfer[%ch1] %res_gpu : !mo.tensor<[D2, D3], f32, gpu:0> to <"cpu", 0>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
