// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// GPU twin of mogg-kernel-api-e2e-trait-tensor-tuple-fused.mlir: the fused
// producers and `trait_tensor_tuple_fused_concat` run on the GPU. The tuple
// holds host-side proxies, so it crosses the launch as a `DevicePassable`,
// as `_MoggDenseTensorArg` does: each element runs its own `_to_device_type`
// into a by-value device tuple, and the device kernel indexes element `k`'s
// access-enabled fused view.

// CHECK: 'trait_tensor_tuple_fused_concat_gpu' returned tensor<5xf32> [2.000000e+00, 4.000000e+00, 2.250000e+00, 6.250000e+00, 1.225000e+01]
mo.graph @trait_tensor_tuple_fused_concat_gpu() -> !mo.tensor<[5], f32> {
  %a = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %b = mo.constant <#M.dense_array<1.5, 2.5, 3.5> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[2], si32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[3], f32> to <"gpu", 0>
  %a2 = mo.add(%a_gpu, %a_gpu) : !mo.tensor<[2], si32, gpu:0>
  %b2 = mo.mul(%b_gpu, %b_gpu) : !mo.tensor<[3], f32, gpu:0>

  %res_gpu = mo.custom [symbol = "trait_tensor_tuple_fused_concat", device = #M.device_ref<"gpu", 0>] (%a2, %b2) : (
    !mo.tensor<[2], si32, gpu:0>,
    !mo.tensor<[3], f32, gpu:0>
  ) -> !mo.tensor<[5], f32, gpu:0>
  %res, %ch3 = mo.transfer[%ch2] %res_gpu : !mo.tensor<[5], f32, gpu:0> to <"cpu", 0>
  mo.output %res : !mo.tensor<[5], f32>
}
