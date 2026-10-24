// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the trait FUSED-LOAD (prologue fusion) emit path on GPU
// (Phase 5 GPU slice). `trait_fused_load_negate` declares its input as
// `Some[Input & Fused & Loadable]`, so the `mo.add` producer is fused into it.
// For a GPU device the graph compiler emits `_MoggInputLoadFusionTensor` with
// `device_passable = True` and `access_enabled = False`: the host proxy can cross
// to the device but cannot load directly. `_to_device_type` projects the fusion
// functor field-by-field (the functor's captured source pointers translate to
// device addresses) and the projected device value has access enabled. The
// kernel's GPU arm binds that projected value by its `Loadable` capability via
// `downcast[XHost.device_type, Loadable]` and runs the fused producer on-device.
//
// The input must live on the GPU for a real device-launch kernel, so the graph
// transfers `%x` to the device (the producer runs there too) and the result
// back -- mirroring the dense `mogg-kernel-api-e2e-trait-gpu.mlir`. With an
// all-ones input, `x.load(i) == x[i] + x[i] == 2`, so the negate kernel returns
// all -2.
// CHECK: 'trait_fused_load_negate_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %p = mo.add(%x_gpu, %x_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.custom [symbol = "trait_fused_load_negate", device = #M.device_ref<"gpu", 0>] (%p) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}

// Same fused-load boundary, but the load is performed by a unified closure that
// captures the PROJECTED load proxy (`downcast[XHost.device_type, Loadable]`).
// The closure is created after projection, so it captures the access-enabled
// `device_type` and the load body's context gate passes; a host-side load fails
// to compile (see the host-negative test). Same `-2` result.
// CHECK: 'trait_fused_load_negate_unified_closure_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_negate_unified_closure_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %p = mo.add(%x_gpu, %x_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.custom [symbol = "trait_fused_load_negate_unified_closure", device = #M.device_ref<"gpu", 0>] (%p) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
