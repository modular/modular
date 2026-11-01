// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the dense CPU trait FUSED-LOAD (prologue fusion) emit
// path (Phase 5): `trait_fused_load_negate` declares its input as
// `Some[Input & Fused & Loadable]`, so the `mo.add` producer is fused into it.
// The graph compiler emits the producer as an `InputFusion` fusion functor and
// wraps it in an inline `_MoggInputLoadFusionArg` (conforming to
// `Input & Fused & Loadable`); the kernel calls `x.load(idx)` to run the fused
// producer. With an all-ones input, `x.load(i) == x[i] + x[i] == 2`, so the
// negate kernel returns all -2 -- proving the emitted fusion wrapper compiles
// and runs with correct numerics through the JIT path.

// CHECK: 'trait_fused_load_negate_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %p = mo.add(%x, %x) : !mo.tensor<[4], f32>
    %z = mo.custom [symbol = "trait_fused_load_negate", device = #M.device_ref<"cpu", 0>] (%p) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
