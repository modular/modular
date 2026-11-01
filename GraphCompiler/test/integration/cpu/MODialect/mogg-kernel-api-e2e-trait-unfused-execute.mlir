// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// `trait_unfused_execute` registers two `execute` overloads: the graph
// compiler calls the fused one when any output has a fused epilogue and the
// plain one when none has. `z` is `x + 1`, which is 2 for an all-ones input.
// `w` records which overload ran: 1 fused, 0 plain.

// Nothing fused: the plain overload writes both outputs through their own
// storage.
// CHECK: 'unfused_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK-NEXT: , tensor<4xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
mo.graph @unfused_graph(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
    %z, %w = mo.custom [symbol = "trait_unfused_execute", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
    mo.output %z, %w : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// A fused epilogue on `z` picks the fused overload for the whole call.
// CHECK: 'negate_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
// CHECK-NEXT: , tensor<4xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @negate_graph(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
    %z, %w = mo.custom [symbol = "trait_unfused_execute", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
    %z2 = mo.negative(%z) : !mo.tensor<[4], f32>
    mo.output %z2, %w : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// Only `w` is fused: the call is still fused, so `z` takes the fused overload's
// identity store and `w` then adds `x` to its record.
// CHECK: 'other_output_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK-NEXT: , tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @other_output_graph(%x: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
    %z, %w = mo.custom [symbol = "trait_unfused_execute", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
    %w2 = mo.add(%w, %x) : !mo.tensor<[4], f32>
    mo.output %z, %w2 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}
