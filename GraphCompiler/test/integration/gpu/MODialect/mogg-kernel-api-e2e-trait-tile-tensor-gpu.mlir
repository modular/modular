// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// CHECK: 'trait_negate_tile_tensor_graph' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_tile_tensor_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_negate_tile_tensor", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
