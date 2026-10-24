// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-pipeline \
// RUN:   --mogg-to-mgp | FileCheck %s

// The body of a side-stream `mo.sequence` may read values captured from the
// enclosing scope. Those reads must chain on the fork wait, not the
// capture's producer chain; main-stream consumers of the same capture are
// unaffected.

// The side-stream view and the fork device_wait exist. The fork is the side
// stream (VIEW) waiting on the captured value's producer chain.
// CHECK: %[[VIEW:[0-9]+]] = mgp.device_context.select_stream[1] %{{[a-z0-9]+}} : !mgp.device_context<<"gpu", 0>>
// CHECK: %[[FORK:[0-9]+]] = mgp.device_wait({{.*}}, {{.*}}, %[[VIEW]], %{{[a-z0-9]+}}) device <"gpu", 0>

// The body's alloc chain includes the fork result; pre-fix it had only the
// capture's producer chain.
// CHECK: %[[BODYCHAIN:[0-9]+]] = grt.chain.create (%[[FORK]], %{{[a-z0-9, ]+}})
// CHECK: mgp.buffer.alloc %[[BODYCHAIN]]

// The main-stream consumer chains on producer + join, never the fork.
// CHECK: %[[JOIN:[0-9]+]] = mgp.device_wait({{.*}}, {{.*}}, %{{[a-z0-9]+}}, %[[VIEW]]) device <"gpu", 0> isSideStreamJoin
// CHECK: grt.chain.create (%{{[a-z0-9]+}}, %[[JOIN]], %{{[a-z0-9]+}})
mo.graph @seqcap(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %prod = mo.relu(%a) : !mo.tensor<[2, 2], f32, gpu:0>
  %s = mo.sequence[1] (%b : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.relu(%prod) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %r : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %main = mo.add(%prod, %b) : !mo.tensor<[2, 2], f32, gpu:0>
  %out = mo.add(%main, %s) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %out : !mo.tensor<[2, 2], f32, gpu:0>
}
