// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --input-shapes=4 --model-inputs=ones | FileCheck %s

// Prologue fusion of a producer with two DISTINCT operands into a trait
// fused-load kernel: `mo.add(%x, %y)` fuses into `trait_fused_load_negate`,
// so the kernel boundary carries two capture arguments for one fused input.
// Each capture derives the fused operand's contract through the bind lambda
// it feeds and is wired to its own buffer capture. This used to crash: the
// positional contract side-table was left one entry short of the argument
// list (GEX-4071).

// CHECK: 'trait_fused_load_two_operands_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_two_operands_graph(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %p = mo.add(%x, %y) : !mo.tensor<[4], f32>
    %z = mo.custom [symbol = "trait_fused_load_negate", device = #M.device_ref<"cpu", 0>] (%p) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
