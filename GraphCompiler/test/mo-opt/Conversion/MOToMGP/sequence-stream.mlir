// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-pipeline \
// RUN:   --mogg-to-mgp | FileCheck %s

// An `mo.sequence` tagged with a non-default stream id lowers by binding its
// body to a `mgp.device_context.select_stream` view of the base device context,
// with a fork `mgp.device_wait` before the body and a join `mgp.device_wait` at
// the first consumer of its results. The body's compiled regions take the VIEW
// context (not the base block arg), which is also what keeps MGP sequence
// fusion from merging them with main-stream work.

// CHECK-LABEL: mgp.model @seq
// The body's device context is a select_stream view of the base gpu:0 context.
// CHECK: %[[VIEW:.+]] = mgp.device_context.select_stream[1] %[[BASE:[a-z0-9]+]] : !mgp.device_context<<"gpu", 0>>
// Fork: the side stream (view) waits before the body runs; it consumes VIEW.
// CHECK: mgp.device_wait{{.*}}%[[VIEW]]
// The body reads the region's input on the side stream.
// CHECK: mgp.tensor.extract.buffer %[[A:[a-z0-9]+]]
// The main-stream relu runs on BASE and does not wait for the side stream.
// CHECK: %[[MAIN:[0-9]+]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}({{.*}}, %[[BASE]])
// Join: the base stream waits on VIEW at the add, the first consumer of the
// body's result, and the join waits for the add's other input first. The host
// thus enqueues the main-stream relu before the wait.
// CHECK: %[[JOININ:[0-9]+]] = grt.chain.create ({{.*}}%[[MAIN]],
// CHECK-NEXT: %[[JOIN:[0-9]+]] = mgp.device_wait(%[[JOININ]], %{{[a-z0-9]+}}, %[[BASE]], %[[VIEW]]) device <"gpu", 0> isSideStreamJoin
// The input stays alive until the join, so the base stream cannot reuse it
// while the side stream still reads it.
// CHECK-NEXT: mgp.extend.lifetime %[[JOIN]], %[[A]] :
// CHECK-NEXT: grt.chain.create ({{.*}}%[[JOIN]],
mo.graph @seq(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %s = mo.sequence[1] (%a : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.relu(%arg) : !mo.tensor<[2, 2], f32, gpu:0>
    %r2 = mo.relu(%r) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %r2 : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %main = mo.relu(%b) : !mo.tensor<[2, 2], f32, gpu:0>
  %out = mo.add(%main, %s) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %out : !mo.tensor<[2, 2], f32, gpu:0>
}

// A replicated (multi-device) body binds one select_stream view per device.

// CHECK-LABEL: mgp.model @seq2dev
// CHECK-DAG: mgp.device_context.select_stream[1] %{{.*}} : !mgp.device_context<<"gpu", 0>>
// CHECK-DAG: mgp.device_context.select_stream[1] %{{.*}} : !mgp.device_context<<"gpu", 1>>
mo.graph @seq2dev(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:1>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) {
  %r:2 = mo.sequence[1] (%a, %b : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) (%x, %y) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) {
    %rx = mo.relu(%x) : !mo.tensor<[2, 2], f32, gpu:0>
    %ry = mo.relu(%y) : !mo.tensor<[2, 2], f32, gpu:1>
    mo.yield %rx, %ry : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>
  }
  mo.output %r#0, %r#1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>
}

// A region input computed in the graph lowers to a tensor view of a buffer the
// body reads. The hold takes that buffer, not the view, which has no other user.

// CHECK-LABEL: mgp.model @seq_computed_input
// CHECK: %[[BUF:[0-9]+]] = mgp.buffer.alloc
// CHECK: mgp.tensor.create %{{[0-9]+}}, %[[BUF]],
// CHECK: mgp.device_context.select_stream[1]
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%{{[0-9]+}}, %[[BUF]],
// CHECK: %[[JOIN:[0-9]+]] = mgp.device_wait({{.*}}isSideStreamJoin
// CHECK-NEXT: mgp.extend.lifetime %[[JOIN]], %[[BUF]] :
mo.graph @seq_computed_input(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %c = mo.relu(%a) : !mo.tensor<[2, 2], f32, gpu:0>
  %s = mo.sequence[1] (%c : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.relu(%arg) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %r : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %main = mo.relu(%b) : !mo.tensor<[2, 2], f32, gpu:0>
  %out = mo.add(%main, %s) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %out : !mo.tensor<[2, 2], f32, gpu:0>
}
