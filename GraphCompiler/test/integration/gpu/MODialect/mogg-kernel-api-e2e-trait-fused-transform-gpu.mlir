// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the trait FUSED-TRANSFORM (compute-output) emit path on
// GPU (Phase 5 GPU slice). `trait_fused_transform_bias` declares its output as
// `Output & Fused & _LegacyComputeOutput` and, with no epilogue fused, refines
// to the compute form in the body. For a GPU device the graph compiler emits the
// `_MoggTransformFusionTensor` wrapper with `device_passable = True` and
// `access_enabled = False`: the host proxy can cross to the device but cannot
// transform or project to a tile directly. It holds the real output buffer as a
// bare `Pointer` (so it translates to a device address; `to_tile_tensor`
// rebuilds the tile) and `_to_device_type` projects the transform functor
// field-by-field. The kernel's GPU arm binds the projected value by
// `downcast[ZHost.device_type, _LegacyComputable & TileTensorable]`, computes
// `x[i] + 1`, applies the fused transform (identity here -- no consumer epilogue
// is fused), and stores through the rebuilt output tile on-device.
//
// The input/output must live on the GPU for a real device-launch kernel, so the
// graph transfers `%x` to the device and the result back. With an all-ones
// input, `z[i] == 1 + 1 == 2`.
// CHECK: 'trait_fused_transform_bias_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_fused_transform_bias_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}

// Same fused-transform boundary, but the transform + final store is performed by
// a unified closure that captures the PROJECTED transform proxy
// (`downcast[ZHost.device_type, _LegacyComputable & TileTensorable]`). The closure is
// created after projection, so it captures the access-enabled `device_type` and
// the transform/to_tile_tensor bodies' context gate passes; a host-side transform
// fails to compile (see the host-negative test). Same `2` result.
// CHECK: 'trait_fused_transform_bias_unified_closure_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_fused_transform_bias_unified_closure_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_fused_transform_bias_unified_closure", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
