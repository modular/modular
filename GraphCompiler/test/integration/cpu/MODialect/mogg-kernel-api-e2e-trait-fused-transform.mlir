// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the dense CPU trait FUSED-TRANSFORM (compute-output) emit
// path (Phase 5): `trait_fused_transform_bias` declares its output as
// `Output & Fused & _LegacyComputeOutput` and, with no epilogue fused, refines
// to the compute form `_LegacyComputable` / `TileTensorable` in the body. The graph compiler
// emits a `TransformFusion` functor (pure value->value) wrapped in an inline
// `_MoggTransformFusionTensor` that composes `_LegacyComputable` (the transform)
// with `TileTensorable` (the real output buffer projection) on ONE operand. The
// kernel computes `x[i] + 1`, applies the fused transform (identity here, since
// no consumer epilogue is fused), and stores the result itself via
// `to_tile_tensor(z)`. With an all-ones input, `z[i] == 1 + 1 == 2` -- proving
// the dual-nature transform wrapper compiles and runs through the JIT path.

// CHECK: 'trait_fused_transform_bias_graph' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @trait_fused_transform_bias_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
