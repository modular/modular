// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the trait FUSED-STORE (epilogue fusion) emit path on GPU
// (Phase 5 GPU slice). `trait_fused_store_double` declares its output as
// `Some[Output & Fused & Storable]`, so the graph compiler emits an
// `_MoggOutputStoreFusionTensor` wrapper; for a GPU device it has
// `device_passable = True` and `access_enabled = False`, so the host proxy can
// cross to the device but cannot store directly. `_to_device_type` projects the
// store functor field-by-field and the projected device value has access enabled.
// The kernel's GPU arm binds that projected value by its `Storable` capability
// via `downcast[ZHost.device_type, Storable]`, computes `2 * x[i]`, and hands it
// to the fused store, which runs the consumer epilogue and writes on-device.
//
// The input/output must live on the GPU for a real device-launch kernel, so the
// graph transfers `%x` to the device and the result back. With an all-ones
// input, `z[i] == 2 * 1 == 2`.
// CHECK: 'trait_fused_store_double_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_fused_store_double_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}

// Same fused-store boundary, but the custom kernel wraps the store operation in
// a host-authored `@__parameter` closure adapter before launching the device
// kernel. The `mo.negative` consumer is the epilogue that should be fused into the
// generated `_MoggOutputStoreFusionTensor`. A constant input of 5 should be
// copied by the custom op, then negated by the fused store.
// CHECK: 'trait_fused_store_parameter_closure_graph' returned tensor<4xf32> [-5.000000e+00, -5.000000e+00, -5.000000e+00, -5.000000e+00]
mo.graph @trait_fused_store_parameter_closure_graph() -> !mo.tensor<[4], f32> {
    %x = mo.constant <#M.dense_array<5.0, 5.0, 5.0, 5.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %raw_gpu = mo.custom [symbol = "trait_fused_store_parameter_closure", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %neg_gpu = mo.negative(%raw_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %neg_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}

// Same fused-store boundary, but the store is performed by a host-created UNIFIED
// closure that CAPTURES the `_MoggOutputStoreFusionTensor` proxy and crosses to
// the device as an explicit runtime argument. The capture is projected through
// `encode_closure_state` -> `_to_device_type`, so the captured proxy's handle is
// translated to a real device address (the discrete-GPU requirement Apple's
// unified memory would hide). The kernel computes `2 * x[i]` and the captured
// fused store applies the `mo.negative` epilogue. A constant input of 5 gives
// `-(2 * 5) == -10`.
// CHECK: 'trait_fused_store_unified_closure_graph' returned tensor<4xf32> [-1.000000e+01, -1.000000e+01, -1.000000e+01, -1.000000e+01]
mo.graph @trait_fused_store_unified_closure_graph() -> !mo.tensor<[4], f32> {
    %x = mo.constant <#M.dense_array<5.0, 5.0, 5.0, 5.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %raw_gpu = mo.custom [symbol = "trait_fused_store_unified_closure", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %neg_gpu = mo.negative(%raw_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %neg_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
