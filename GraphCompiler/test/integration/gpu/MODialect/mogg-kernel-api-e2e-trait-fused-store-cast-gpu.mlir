// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// GPU arm of the fused cast epilogue. The kernel stores f32 values through
// the device proxy, and the fusion functor casts and writes the si32 buffer.
// With all-ones input the kernel writes 2, which casts to 2.
// CHECK: 'trait_fused_store_cast_graph' returned tensor<4xsi32> [2, 2, 2, 2]
mo.graph @trait_fused_store_cast_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], si32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %d_gpu = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.cast(%d_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], si32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], si32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], si32>
}
