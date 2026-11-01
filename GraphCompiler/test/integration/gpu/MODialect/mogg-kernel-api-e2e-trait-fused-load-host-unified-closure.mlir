// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the GPU trait fused-load path with a host-built UNIFIED
// closure. The closure is built inside `execute` on the host, captures the
// access-enabled view returned by `unsafe_fused_view()`, and is threaded as a
// value through `_fused_load_negate`, which re-captures it into the device
// kernel body it launches. With an all-ones input fused through `mo.add`, the
// negate kernel returns all -2.

// CHECK: 'trait_fused_load_host_unified_closure_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_host_unified_closure_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %p = mo.add(%x_gpu, %x_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.custom [symbol = "trait_fused_load_host_unified_closure", device = #M.device_ref<"gpu", 0>] (%p) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
