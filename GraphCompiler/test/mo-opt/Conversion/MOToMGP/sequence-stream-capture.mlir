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

// A body that reads none of its inputs and orders its side effects only by a
// chain captured from the enclosing scope: a host-gated copy into a persistent
// buffer. The wait and the copy start after the fork, and the join keeps the
// fork alive; otherwise they run on the side stream before it joins a device
// graph capture, and the capture rejects the join.

// CHECK-LABEL: mgp.model @seq_chained
// CHECK: %[[VIEW:[0-9]+]] = mgp.device_context.select_stream[2] %[[BASE:[a-z0-9]+]] : !mgp.device_context<<"gpu", 0>>
// CHECK: %[[FORK:[0-9]+]] = mgp.device_wait({{.*}}, {{.*}}, %[[VIEW]], %[[BASE]]) device <"gpu", 0> isSideStreamFork
// CHECK: %[[BODYCHAIN:[0-9]+]] = grt.chain.create (%{{[0-9]+}}, %[[FORK]])
// CHECK: %[[WAIT:[0-9]+]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[BODYCHAIN]], {{.*}}, %[[VIEW]])
// CHECK: mogg.call @__wait_host_value_with_dep
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[WAIT]], {{.*}}, %[[VIEW]])
// CHECK: mogg.call @__inplace_memcpy
// CHECK: %[[JOININ:[0-9]+]] = grt.chain.create (%[[FORK]], %{{[0-9]+}})
// CHECK-NEXT: mgp.device_wait(%[[JOININ]], %{{[a-z0-9]+}}, %[[BASE]], %[[VIEW]]) device <"gpu", 0> isSideStreamJoin
mo.graph @seq_chained(%flag: !mo.buffer<[2], si64>, %scratch: !mo.buffer<[4, 8], si32, gpu:0>, %pinned: !mo.tensor<[4, 8], si32>, %x: !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[4, 8], si32, gpu:0>) {
  %c = mo.chain.create()
  %s = mo.sequence[2] (%x : !mo.tensor<[2, 2], f32, gpu:0>) (%unused) -> (!mo.tensor<[4, 8], si32, gpu:0>) {
    %w = mo.custom [symbol = "mo.wait_host_value_with_dep", device = #M.device_ref<"gpu", 0>](%flag, %scratch, %c) : (!mo.buffer<[2], si64>, !mo.buffer<[4, 8], si32, gpu:0>, !mo.chain) -> !mo.chain
    %m = mo.custom [symbol = "mo.inplace_memcpy", device = #M.device_ref<"gpu", 0>, parameters = {DstDevice = "gpu", SrcDevice = "cpu"}](%scratch, %pinned, %w) : (!mo.buffer<[4, 8], si32, gpu:0>, !mo.tensor<[4, 8], si32>, !mo.chain) -> !mo.chain
    %t, %tc = mo.mutable.load(%scratch, %m) : (!mo.buffer<[4, 8], si32, gpu:0>, !mo.chain) -> (!mo.tensor<[4, 8], si32, gpu:0>, !mo.chain)
    mo.yield %t : !mo.tensor<[4, 8], si32, gpu:0>
  }
  %main = mo.relu(%x) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %main, %s : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[4, 8], si32, gpu:0>
}
