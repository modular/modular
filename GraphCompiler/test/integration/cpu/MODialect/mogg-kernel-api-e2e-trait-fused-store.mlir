// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the dense CPU trait FUSED-STORE (epilogue fusion) emit
// path (Phase 5): `trait_fused_store_double` declares its output as
// `Some[Output & Fused & Storable]`, so the graph compiler emits an
// `OutputStoreFusion` fusion functor (capturing the output's MUTABLE
// `TileTensor` projection) wrapped in an inline `_MoggOutputStoreFusionTensor`
// (conforming to `Output & Fused & Storable`); the kernel calls
// `z.store(idx, 2 * x[i])` and the functor writes through the captured tile.
// With an all-ones input, `z[i] == 2 * 1 == 2` -- proving the emitted store
// fusion wrapper compiles and runs with correct numerics through the JIT path.

// CHECK: 'trait_fused_store_double_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_fused_store_double_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_fused_store_double", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
