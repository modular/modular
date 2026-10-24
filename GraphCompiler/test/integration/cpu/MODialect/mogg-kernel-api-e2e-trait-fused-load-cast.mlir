// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// Fused cast prologue: the kernel sees an f32 proxy, while the fusion
// functor loads si32 and casts. All-ones si32 input becomes 1.0, the kernel
// negates it, so the result is all -1.
// CHECK: 'trait_fused_load_cast_graph' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_fused_load_cast_graph(%x: !mo.tensor<[4], si32>) -> !mo.tensor<[4], f32> {
    %p = mo.cast(%x) : (!mo.tensor<[4], si32>) -> !mo.tensor<[4], f32>
    %z = mo.custom [symbol = "trait_fused_load_negate", device = #M.device_ref<"cpu", 0>] (%p) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
