// RUN: mo-opt %s | FileCheck %s

// Compile-time sniff test: 3 ops that share no shape or dtype, so the
// kernel-stub dedup pass (MergeIdenticalFuncs) has nothing to merge --
// the "no caching" control case. Dynamic leading dimension `D0` throughout
// -- see `small-distinct-ops-static.mlir` for the all-static sibling.
// See README.md in this directory.

// CHECK: mo.graph @distinct_ops_dynamic
mo.graph @distinct_ops_dynamic<D0>(
    %a: !mo.tensor<[D0], f32, gpu:0>,
    %b: !mo.tensor<[D0], f32, gpu:0>,
    %c: !mo.tensor<[D0, 8], f32, gpu:0>,
    %d: !mo.tensor<[D0, 8], f32, gpu:0>,
    %e: !mo.tensor<[D0], bf16, gpu:0>,
    %f: !mo.tensor<[D0], bf16, gpu:0>
) -> (!mo.tensor<[D0], f32, gpu:0>, !mo.tensor<[D0, 8], f32, gpu:0>, !mo.tensor<[D0], bf16, gpu:0>) {
  %0 = mo.add(%a, %b) : !mo.tensor<[D0], f32, gpu:0>
  %1 = mo.mul(%c, %d) : !mo.tensor<[D0, 8], f32, gpu:0>
  %2 = mo.add(%e, %f) : !mo.tensor<[D0], bf16, gpu:0>
  mo.output %0, %1, %2 : !mo.tensor<[D0], f32, gpu:0>, !mo.tensor<[D0, 8], f32, gpu:0>, !mo.tensor<[D0], bf16, gpu:0>
}
