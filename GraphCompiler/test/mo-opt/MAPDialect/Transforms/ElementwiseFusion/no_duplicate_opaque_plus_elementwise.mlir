// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// Regression guard for GEX-4228: a cheap multi-use producer that feeds an opaque
// consumer (through a prologue-capable input) AND an elementwise foreach at the
// same time must NOT be duplicated into each. The opaque materializes the value
// anyway, so cloning the producer into the foreach as well is pure
// recomputation -- which is what blew up residual chains. It is materialized
// once and both consumers read it, matching the legacy elementwise fuser.
//
// Here `exp(%a)` feeds `gather` (opaque, prologue-capable) and `tanh` (foreach).

mo.graph @no_dup_opaque_plus_elementwise(
  %a: !mo.tensor<[4096, 2], f32, gpu:0>,
  %indices: !mo.tensor<[2, 2], si32, gpu:0>
) -> (!mo.tensor<[2, 2, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%a) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.gather(%0, %indices) [axis = 0] : (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[2, 2], si32, gpu:0>) -> !mo.tensor<[2, 2, 2], f32, gpu:0>
  %2 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[2, 2, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// The producer is materialized once, not cloned into the gather prologue AND the
// tanh foreach: exactly one `Exp` call survives.
// CHECK-LABEL: mo.graph @no_dup_opaque_plus_elementwise
// CHECK-COUNT-1: map.call.elementwise @{{.*}}.Exp
// CHECK-NOT:     map.call.elementwise @{{.*}}.Exp
