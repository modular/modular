// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=2x3 --model-inputs=ones | FileCheck %s

// The device path must preserve the runtime extent through `_to_device_type`;
// explicit transfers make this a real GPU launch.
// CHECK: 'trait_negate_dynamic' returned tensor<2x3xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_dynamic<D0>(%x: !mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[2, D0], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_negate", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[2, D0], f32, gpu:0>) -> !mo.tensor<[2, D0], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[2, D0], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[2, D0], f32>
}
