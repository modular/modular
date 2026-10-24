// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// `y` is a trait-bound dense input backed by a folded constant, while `x`
// remains a boundary input. Both must assemble into values accepted by the
// same kernel call.

// CHECK: 'trait_constant_add_graph' returned tensor<4xf32> [2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]
mo.graph @trait_constant_add_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %cst = mo.constant <#M.dense_array<1., 2., 3., 4.> : tensor<4xf32>> : !mo.tensor<[4], f32>
    %z = mo.custom [symbol = "trait_mixed_legacy_out_add", device = #M.device_ref<"cpu", 0>] (%x, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
