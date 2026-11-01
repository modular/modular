// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// A trait kernel called with the same value twice: `SimplifyKernelOperands`
// merges the duplicate operands into one exec-body argument that feeds two
// `execute` operand positions. Both positions declare the same
// (`input`/dense) contract, so the derived contract is unambiguous and the
// kernel runs off a single buffer capture. This used to crash: the positional
// contract side-table kept both entries while the argument list shrank
// (GEX-4071).

// CHECK: 'trait_dup_operand_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_dup_operand_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_mixed_legacy_out_add", device = #M.device_ref<"cpu", 0>] (%x, %x) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
