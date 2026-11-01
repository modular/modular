// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// Fused cast epilogue: the kernel stores f32 values, and the fusion functor
// casts and writes the si32 buffer. With all-ones input the kernel writes 2,
// which casts to 2.
// CHECK: 'trait_fused_store_cast_graph' returned tensor<4xsi32> [2, 2, 2, 2]
mo.graph @trait_fused_store_cast_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], si32> {
    %d = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %z = mo.cast(%d) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], si32>
    mo.output %z : !mo.tensor<[4], si32>
}
