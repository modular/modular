// RUN: mo-opt %s | FileCheck %s

// Compile-time sniff test: two separate mo.graphs (matmul + broadcast
// bias-add + relu) with identical shapes, sharing the same symbolic leading
// dimension `D0`. Kernel-stub dedup (MergeIdenticalFuncs) runs on the
// whole module, not per-graph, so this is the "dedup across mo.graph ops"
// case, exercised with dynamic shapes -- both graphs' kernels are expected
// to collapse onto shared ones. See `medium-duplicate-mlp-graphs-static.mlir`
// for the all-static sibling. See README.md in this directory.

// CHECK-DAG: mo.graph @mlp_a_dynamic
// CHECK-DAG: mo.graph @mlp_b_dynamic

mo.graph @mlp_a_dynamic<D0>(%a: !mo.tensor<[D0, 256], bf16, gpu:0>, %w: !mo.tensor<[256, 256], bf16, gpu:0>, %bias: !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0> {
  %0 = mo.matmul(%a, %w) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %1 = mo.static.broadcast_to(%bias) : !mo.tensor<[256], bf16, gpu:0> -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[D0, 256], bf16, gpu:0>
  %3 = mo.relu(%2) : !mo.tensor<[D0, 256], bf16, gpu:0>
  mo.output %3 : !mo.tensor<[D0, 256], bf16, gpu:0>
}

mo.graph @mlp_b_dynamic<D0>(%a: !mo.tensor<[D0, 256], bf16, gpu:0>, %w: !mo.tensor<[256, 256], bf16, gpu:0>, %bias: !mo.tensor<[256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0> {
  %0 = mo.matmul(%a, %w) : (!mo.tensor<[D0, 256], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>) -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %1 = mo.static.broadcast_to(%bias) : !mo.tensor<[256], bf16, gpu:0> -> !mo.tensor<[D0, 256], bf16, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[D0, 256], bf16, gpu:0>
  %3 = mo.relu(%2) : !mo.tensor<[D0, 256], bf16, gpu:0>
  mo.output %3 : !mo.tensor<[D0, 256], bf16, gpu:0>
}
