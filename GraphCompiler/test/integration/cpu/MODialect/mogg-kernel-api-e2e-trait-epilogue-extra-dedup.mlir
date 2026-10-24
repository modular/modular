// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// Epilogue fusion whose extra consumer operand is the producer's own input:
// one kernel argument feeds both the dense input execute position and the
// epilogue capture. The direct operand position wins the contract derivation
// (dense input); the capture only cross-checks the lowering lane. This used to
// hard-error with a contract conflict (GEX-4071).

// CHECK: 'trait_epilogue_extra_dedup_graph' returned tensor<4xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @trait_epilogue_extra_dedup_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %w = mo.add(%z, %x) : !mo.tensor<[4], f32>
    mo.output %w : !mo.tensor<[4], f32>
}
