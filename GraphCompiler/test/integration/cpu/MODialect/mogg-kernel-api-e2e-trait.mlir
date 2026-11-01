// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the dense CPU trait-argument emit path (Slice 3 /
// Phase 4): `trait_negate`'s arguments carry the minimal graph-visible markers
// (`Some[Output & TileTensorable]` / `Some[Input & TileTensorable]`) and refine
// `TileTensorable` in the body, so the graph compiler emits an inline
// `_MoggDenseTensorArg` for each argument and infers the kernel's generic `T`
// from it. With an all-ones input the negate kernel must
// return all -1, proving the emitted Mojo compiles and runs with correct
// numerics through the JIT path.

// CHECK: 'trait_negate_graph' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_negate", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
