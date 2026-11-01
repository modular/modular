// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// GPU arm of mogg-kernel-api-e2e-trait-fused-transform-epilogue.mlir.
// `trait_fused_transform_bias` computes `x[i] + 1`, which is 2 for an all-ones
// input, and launches the compute-form or store-form device kernel depending on
// the form its `_LegacyComputeOutput` slot arrives in.

// A same-dtype elementwise epilogue arrives in the compute form.
// CHECK: 'trait_fused_transform_negate_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_transform_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %out_gpu = mo.negative(%z_gpu) : !mo.tensor<[4], f32, gpu:0>
    %out, %ch2 = mo.transfer[%ch1] %out_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %out : !mo.tensor<[4], f32>
}

// A cast epilogue is downgraded to the store form.
// CHECK: 'trait_fused_transform_cast_graph' returned tensor<4xsi32> [2, 2, 2, 2]
mo.graph @trait_fused_transform_cast_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], si32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %out_gpu = mo.cast(%z_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], si32, gpu:0>
    %out, %ch2 = mo.transfer[%ch1] %out_gpu : !mo.tensor<[4], si32, gpu:0> to <"cpu", 0>
    mo.output %out : !mo.tensor<[4], si32>
}

// A reshape epilogue is downgraded to the store form.
// CHECK: 'trait_fused_transform_reshape_graph' returned tensor<2x2xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_transform_reshape_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %r_gpu = mo.static.reshape(%z_gpu) : !mo.tensor<[4], f32, gpu:0> -> !mo.tensor<[2, 2], f32, gpu:0>
    %out_gpu = mo.negative(%r_gpu) : !mo.tensor<[2, 2], f32, gpu:0>
    %out, %ch2 = mo.transfer[%ch1] %out_gpu : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
    mo.output %out : !mo.tensor<[2, 2], f32>
}
