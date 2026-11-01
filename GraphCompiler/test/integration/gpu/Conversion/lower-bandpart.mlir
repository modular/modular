
// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s -o %t.mlir

// RUN: cat %t.mlir | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --input-shapes=3x2 --result-output-style=full %t.mlir | FileCheck --check-prefix=CHECK-MT %s

// CHECK-IR-LABEL:  mef.callable @linalg_band_part_rank_2_dyn$exec
// CHECK-IR: "mgp.generic.execute"{{.*}}linalg.band_part

mo.graph @linalg_band_part_rank_2_dyn<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %3 = mo.constant <#M.dense_array<0> : tensor<ui8>> : !mo.tensor<[], bool>
  %ch0 = mo.chain.create()
  %arg0_gpu, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[D0, D1], f32> to <"gpu", 0>
  %4 = mo.linalg.band_part(%arg0_gpu, %1, %2, %3) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[D0, D1], f32, gpu:0>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1], f32, gpu:0>
  %5, %ch2 = mo.transfer[%ch1] %4 : !mo.tensor<[D0, D1], f32, gpu:0> to <"cpu", 0>
  mo.output %5 : !mo.tensor<[D0, D1], f32>
}
// CHECK-MT: 'linalg_band_part_rank_2_dyn' returned tensor<3x2xf32> [1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00
