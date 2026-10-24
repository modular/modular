// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of a MIXED-LANE kernel: `trait_mixed_legacy_add`'s output
// carries the composable tensor-argument traits (`Some[Output &
// TileTensorable]`), so the graph compiler emits an inline `_MoggDenseTensorArg`
// for it, while its two inputs are legacy `InputTensor` (`ManagedTensorSlice`)
// operands, unchanged. `CallExecuteOp::emitMojo` must assemble both lanes'
// captures in the same call. With all-ones inputs, `z[i] = x[i] + y[i]` must
// return all 2, proving the two lanes compose and the emitted Mojo compiles
// and runs with correct numerics -- a gap the lowering-only FileCheck in
// infra-lower-tensor-arg-traits.mlir does not cover.

// CHECK: 'trait_mixed_legacy_add_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_mixed_legacy_add_graph(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_mixed_legacy_add", device = #M.device_ref<"cpu", 0>] (%x, %y) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
