// RUN: mo-opt %s --verify-diagnostics

// Sentinel: today's `mo.parallel` verifier rejects bundles whose element
// tensors carry distinct dim parameters. Unlocking heterogeneous-launch
// parallel ops requires additional work:
//
//   1. Relax `ParallelOp::verify` to accept heterogeneous element dims.
//   2. Substitute parameter names per launch in `lowerParallelOp::clone`,
//      not just SSA values.
//   3. Translate canonical kernel-local names to per-launch graph-level
//      `KGEN::ParamDeclRefAttr`s when emitting each call site.

mo.graph @parallel_distinct_params<D0, D1>(
    %arg0: !mo.tensor<[D0, 3], f32>,
    %arg1: !mo.tensor<[D1, 3], f32>)
    -> (!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>) {
  %bundle = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>)
      -> (!mo.bundle<[!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>]>)
  // expected-error @+1 {{'mo.parallel' op bundle input #0 element #1 type '!mo.tensor<[D1, 3], f32>' does not match block argument type '!mo.tensor<[D0, 3], f32>'}}
  %res_bundle = mo.parallel (%x) in
      (%bundle : !mo.bundle<[!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>]>) {
    %1 = mo.relu(%x) : !mo.tensor<[D0, 3], f32>
    mo.yield %1 : !mo.tensor<[D0, 3], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle)
      : (!mo.bundle<[!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>]>)
      -> (!mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>)
  mo.output %res_0, %res_1
      : !mo.tensor<[D0, 3], f32>, !mo.tensor<[D1, 3], f32>
}
