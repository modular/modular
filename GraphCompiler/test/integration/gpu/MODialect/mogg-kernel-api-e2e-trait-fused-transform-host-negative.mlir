// RUN: not mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect 2>&1 | FileCheck %s

// NEGATIVE test (expected to FAIL compilation). `trait_fused_transform_host_negative`
// transforms DIRECTLY on the fused-transform proxy at the host entry, with no
// projection. On a GPU device the graph compiler emits the device-passable HOST
// proxy (`_MoggTransformFusionTensor` with `access_enabled = False`), so the
// transform body's context gate -- `comptime assert Self._access_enabled` --
// rejects it at instantiation. `mo-opt` must error out instead of lowering.
// CHECK: fused transform is only valid on an access-enabled tensor
mo.graph @trait_fused_transform_host_negative_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_host_negative", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
