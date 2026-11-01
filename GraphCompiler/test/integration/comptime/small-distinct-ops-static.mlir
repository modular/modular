// RUN: mo-opt %s | FileCheck %s

// Compile-time sniff test: 3 ops that share no shape or dtype, so the
// kernel-stub dedup pass (MergeIdenticalFuncs) has nothing to merge --
// the "no caching" control case. Static shapes throughout -- see
// `small-distinct-ops-dynamic.mlir` for the dynamic-leading-dim sibling.
// See README.md in this directory.

// CHECK: mo.graph @distinct_ops
mo.graph @distinct_ops(
    %a: !mo.tensor<[4], f32, gpu:0>,
    %b: !mo.tensor<[4], f32, gpu:0>,
    %c: !mo.tensor<[4, 8], f32, gpu:0>,
    %d: !mo.tensor<[4, 8], f32, gpu:0>,
    %e: !mo.tensor<[16], bf16, gpu:0>,
    %f: !mo.tensor<[16], bf16, gpu:0>
) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[16], bf16, gpu:0>) {
  %0 = mo.add(%a, %b) : !mo.tensor<[4], f32, gpu:0>
  %1 = mo.mul(%c, %d) : !mo.tensor<[4, 8], f32, gpu:0>
  %2 = mo.add(%e, %f) : !mo.tensor<[16], bf16, gpu:0>
  mo.output %0, %1, %2 : !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[16], bf16, gpu:0>
}
