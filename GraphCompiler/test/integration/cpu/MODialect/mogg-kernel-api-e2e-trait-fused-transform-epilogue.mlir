// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// `trait_fused_transform_bias` declares its output as a `_LegacyComputeOutput`
// slot and computes `x[i] + 1`, which is 2 for an all-ones input. Each graph
// fuses a different epilogue into it, so the kernel runs in each form. See
// Conversion/MAPToMOGG/trait-compute-output-slot.mlir for the classification.

// A same-dtype elementwise epilogue arrives in the compute form: the kernel
// calls `compute` and stores the result itself.
// CHECK: 'trait_fused_transform_negate_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_transform_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %out = mo.negative(%z) : !mo.tensor<[4], f32>
    mo.output %out : !mo.tensor<[4], f32>
}

// A cast epilogue is downgraded to the store form: the kernel stores the f32
// it computed, and the epilogue casts and writes the si32 buffer.
// CHECK: 'trait_fused_transform_cast_graph' returned tensor<4xsi32> [2, 2, 2, 2]
mo.graph @trait_fused_transform_cast_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], si32> {
    %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %out = mo.cast(%z) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], si32>
    mo.output %out : !mo.tensor<[4], si32>
}

// A reshape epilogue is downgraded to the store form: the epilogue writes
// each value at the reshaped index.
// CHECK: 'trait_fused_transform_reshape_graph' returned tensor<2x2xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_transform_reshape_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
    %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %r = mo.static.reshape(%z) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 2], f32>
    %out = mo.negative(%r) : !mo.tensor<[2, 2], f32>
    mo.output %out : !mo.tensor<[2, 2], f32>
}

// A fused-output `TensorTuple` declared before the slot: tuple element 1's
// epilogue stays a store lambda although `z` sits at its raw result position,
// and `z`'s epilogue still arrives in the compute form.
// CHECK: 'trait_tuple_before_compute_output_graph' returned tensor<4xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
// CHECK-NEXT: , tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK-NEXT: , tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_tuple_before_compute_output_graph(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
    %a, %b, %z = mo.custom [symbol = "trait_tuple_then_compute_output", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
    %b2 = mo.add(%b, %x) : !mo.tensor<[4], f32>
    %z2 = mo.negative(%z) : !mo.tensor<[4], f32>
    mo.output %a, %b2, %z2 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}
