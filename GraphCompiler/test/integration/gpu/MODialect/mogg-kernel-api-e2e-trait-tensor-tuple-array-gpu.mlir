// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// `trait_tensor_tuple_add_array`: the GPU cases of
// mogg-kernel-api-e2e-trait-tensor-tuple-gpu.mlir, with the kernel converting
// each dense tuple to an `Array` of dynamic-layout `TileTensor`s on the host, as
// `mo.split` does. Each side's elements share one type, so every case uses one
// dtype per side.

// CHECK: 'trait_tensor_tuple_add_array_size3_si32_gpu' returned tensor<2xsi32> [1001, 1002]
// CHECK-NEXT: , tensor<3xsi32> [1010, 1020, 1030]
// CHECK-NEXT: , tensor<1xsi32> [1100]
mo.graph @trait_tensor_tuple_add_array_size3_si32_gpu() -> (!mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[1], si32>) {
  %in1 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %in2 = mo.constant <#M.dense_array<10, 20, 30> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %in3 = mo.constant <#M.dense_array<100> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %bias = mo.constant <#M.dense_array<1000> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %ch0 = mo.chain.create()
  %in1_gpu, %ch1 = mo.transfer[%ch0] %in1 : !mo.tensor<[2], si32> to <"gpu", 0>
  %in2_gpu, %ch2 = mo.transfer[%ch1] %in2 : !mo.tensor<[3], si32> to <"gpu", 0>
  %in3_gpu, %ch3 = mo.transfer[%ch2] %in3 : !mo.tensor<[1], si32> to <"gpu", 0>
  %bias_gpu, %ch4 = mo.transfer[%ch3] %bias : !mo.tensor<[1], si32> to <"gpu", 0>

  %out1_gpu, %out2_gpu, %out3_gpu = mo.custom [symbol = "trait_tensor_tuple_add_array", device = #M.device_ref<"gpu", 0>] (%bias_gpu, %in1_gpu, %in2_gpu, %in3_gpu) : (
    !mo.tensor<[1], si32, gpu:0>,
    !mo.tensor<[2], si32, gpu:0>,
    !mo.tensor<[3], si32, gpu:0>,
    !mo.tensor<[1], si32, gpu:0>
  ) -> (!mo.tensor<[2], si32, gpu:0>, !mo.tensor<[3], si32, gpu:0>, !mo.tensor<[1], si32, gpu:0>)
  %out1, %ch5 = mo.transfer[%ch4] %out1_gpu : !mo.tensor<[2], si32, gpu:0> to <"cpu", 0>
  %out2, %ch6 = mo.transfer[%ch5] %out2_gpu : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  %out3, %ch7 = mo.transfer[%ch6] %out3_gpu : !mo.tensor<[1], si32, gpu:0> to <"cpu", 0>
  mo.output %out1, %out2, %out3 : !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[1], si32>
}

// One-element tuples bind exactly as single trait arguments would.
// CHECK: 'trait_tensor_tuple_add_array_size1_f32_gpu' returned tensor<2xf32> [8.000000e+00, 9.000000e+00]
mo.graph @trait_tensor_tuple_add_array_size1_f32_gpu() -> !mo.tensor<[2], f32> {
  %in1 = mo.constant <#M.dense_array<7.0, 8.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %bias = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %ch0 = mo.chain.create()
  %in1_gpu, %ch1 = mo.transfer[%ch0] %in1 : !mo.tensor<[2], f32> to <"gpu", 0>
  %bias_gpu, %ch2 = mo.transfer[%ch1] %bias : !mo.tensor<[1], f32> to <"gpu", 0>

  %res_gpu = mo.custom [symbol = "trait_tensor_tuple_add_array", device = #M.device_ref<"gpu", 0>] (%bias_gpu, %in1_gpu) : (
    !mo.tensor<[1], f32, gpu:0>,
    !mo.tensor<[2], f32, gpu:0>
  ) -> !mo.tensor<[2], f32, gpu:0>
  %res, %ch3 = mo.transfer[%ch2] %res_gpu : !mo.tensor<[2], f32, gpu:0> to <"cpu", 0>
  mo.output %res : !mo.tensor<[2], f32>
}
