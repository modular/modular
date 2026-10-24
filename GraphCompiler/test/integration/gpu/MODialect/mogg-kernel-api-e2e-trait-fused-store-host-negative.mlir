// RUN: not mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect 2>&1 | FileCheck %s

// NEGATIVE test (expected to FAIL compilation). `trait_fused_store_host_negative`
// stores DIRECTLY on the fused-store proxy from the host entry, with no
// projection. On a GPU device the graph compiler emits the device-passable HOST
// proxy (`_MoggOutputStoreFusionTensor` with `access_enabled = False`), so the
// store body's context gate -- `comptime assert Self._access_enabled` -- rejects
// the store at instantiation. This is the host-side-store footgun the gate exists
// to prevent, so `mo-opt` must error out instead of lowering.
// CHECK: only valid on an access-enabled tensor
mo.graph @trait_fused_store_host_negative_graph() -> !mo.tensor<[4], f32> {
    %x = mo.constant <#M.dense_array<5.0, 5.0, 5.0, 5.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %raw_gpu = mo.custom [symbol = "trait_fused_store_host_negative", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %neg_gpu = mo.negative(%raw_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %neg_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
