// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline \
// RUN:   --mogg-to-mgp | FileCheck %s

// `mo.stable.create` has no kernel: it survives kernel lowering and becomes
// `mgp.device_graph.stable.create` over the input's lowered tensor. The
// marker travels with the type, so the graph result type converts to a stable
// `!mgp.tensor` too.

// CHECK-LABEL: mgp.model @stable_create
// CHECK: %[[S:.+]] = mgp.device_graph.stable.create %{{.+}} : !mgp.tensor<4xf32, <"gpu", 0>>
// CHECK: mef.output %{{.*}}, %[[S]] : !mef.chain, !mgp.tensor<4xf32, <"gpu", 0>, stable>
mo.graph @stable_create(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0, {stable}> {
  %0 = mo.stable.create %arg0 : !mo.tensor<[4], f32, gpu:0, {stable}>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0, {stable}>
}

// A symbolic dimension lowers to a runtime dim operand on the MGP op, taken
// from the graph's dimension parameter rather than the input's own spec.
// CHECK-LABEL: mgp.model @stable_create_dynamic
// CHECK: %[[S:.+]] = mgp.device_graph.stable.create %{{.+}}[%{{.+}}] : !mgp.tensor<?xf32, <"gpu", 0>>
// CHECK: mef.output %{{.*}}, %[[S]] : !mef.chain, !mgp.tensor<?xf32, <"gpu", 0>, stable>
mo.graph @stable_create_dynamic<N>(%arg0: !mo.tensor<[N], f32, gpu:0>) -> !mo.tensor<[N], f32, gpu:0, {stable}> {
  %0 = mo.stable.create %arg0 : !mo.tensor<[N], f32, gpu:0, {stable}>
  mo.output %0 : !mo.tensor<[N], f32, gpu:0, {stable}>
}
