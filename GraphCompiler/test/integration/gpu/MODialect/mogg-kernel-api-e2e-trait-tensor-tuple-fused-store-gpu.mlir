// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// GPU twin of mogg-kernel-api-e2e-trait-tensor-tuple-fused-store.mlir:
// `trait_tensor_tuple_fused_store_split` and its fused epilogues run on the
// GPU. The tuple of `_MoggOutputStoreFusionTensor`s crosses the launch as a
// `DevicePassable`, and the device kernel stores through element `k`'s
// access-enabled view, which runs that output's epilogue on-device.

// CHECK: 'trait_tensor_tuple_fused_store_split_gpu' returned tensor<2xf32> [1.100000e+01, 2.200000e+01]
// CHECK-NEXT: , tensor<3xf32> [3.000000e+00, 4.000000e+00, 5.000000e+00]
mo.graph @trait_tensor_tuple_fused_store_split_gpu() -> (!mo.tensor<[2], f32>, !mo.tensor<[3], f32>) {
  %x = mo.constant <#M.dense_array<1.5, 2.5, 3.5, 4.5, 5.5> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %c = mo.constant <#M.dense_array<9.5, 19.5> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %ch0 = mo.chain.create()
  %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[5], f32> to <"gpu", 0>
  %c_gpu, %ch2 = mo.transfer[%ch1] %c : !mo.tensor<[2], f32> to <"gpu", 0>

  %a, %b = mo.custom [symbol = "trait_tensor_tuple_fused_store_split", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (
    !mo.tensor<[5], f32, gpu:0>
  ) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], si32, gpu:0>)
  %a2 = mo.add(%a, %c_gpu) : !mo.tensor<[2], f32, gpu:0>
  %b2 = mo.cast(%b) : (!mo.tensor<[3], si32, gpu:0>) -> !mo.tensor<[3], f32, gpu:0>
  %a_cpu, %ch3 = mo.transfer[%ch2] %a2 : !mo.tensor<[2], f32, gpu:0> to <"cpu", 0>
  %b_cpu, %ch4 = mo.transfer[%ch3] %b2 : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  mo.output %a_cpu, %b_cpu : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>
}
