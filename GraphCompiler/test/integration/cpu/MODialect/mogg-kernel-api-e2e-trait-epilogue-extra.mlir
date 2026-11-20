// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --input-shapes=4 --model-inputs=ones | FileCheck %s

// Epilogue fusion of `mo.add` into a trait fused-store producer with an EXTRA
// consumer operand: `%y` reaches the fused kernel only as a captured input of
// the epilogue, so it lowers as a trait buffer capture read by the store
// functor rather than a distinct kernel input (GEX-4071). The end-to-end result
// pins that the capture is wired correctly.

// CHECK: 'trait_epilogue_extra_graph' returned tensor<4xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @trait_epilogue_extra_graph(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %w = mo.add(%z, %y) : !mo.tensor<[4], f32>
    mo.output %w : !mo.tensor<[4], f32>
}
