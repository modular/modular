// RUN: not mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect 2>&1 | FileCheck %s

// NEGATIVE control: a host-built closure that captures the fused-load host
// proxy (`_access_enabled = False`) directly must fail at launch. The closure
// body is compiled on the host before projection, so its load reaches the
// access-disabled proxy and trips the gate. The gate keys on what is captured,
// not on the closure flavor or how many layers it is threaded through.
// Positive counterpart:
// `mogg-kernel-api-e2e-trait-fused-load-host-unified-closure.mlir`.
// CHECK: fused load is only valid on an access-enabled tensor
mo.graph @trait_fused_load_host_closure_negative_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %p = mo.add(%x_gpu, %x_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.custom [symbol = "trait_fused_load_host_closure_negative", device = #M.device_ref<"gpu", 0>] (%p) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
