// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize)), map-to-mogg)' | FileCheck %s

// Full pipeline from a real mo.static.broadcast_to op through MOToMAP,
// outlining, view motion, and MAPToMOGG: map.index.broadcast.stride lowers to
// mogg.tensor.create.broadcast, which only carries the view's own output
// shape -- the input's shape/rank is read off the input tensor operand at
// Mojo-emission time, not duplicated onto this op. The `canonicalize` right
// after LoopInvariantViewMotion drops the closure's now-dead original input
// capture (unused once the hoisted broadcast view replaces the load) via
// `DropDeadCapturingClosureCaptures`.

// Rank-preserving: in-shape dim 0 (size 1) broadcasts to 8.

// CHECK-LABEL: mo.graph @broadcastSameRank
// CHECK:       [[KERNEL:%[^ ]+]] = mogg.routine ({{.*}} : !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]])
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]])
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[1, 64]> : !mosh.ape>
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg.tensor.create([[IP]], [[ISH]]) {{.*}}io = Input{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8, 64]> : !mosh.ape>
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) {{.*}}io = Output{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[BCAST:%[^ ]+]] = mogg.tensor.create.broadcast([[IT]]) {{.*}}outShape = #mosh<ape[8, 64]>{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8, 64]> : !mosh.ape>
// CHECK-NEXT:    mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) ([[IDX:%[^ :,]+]] : !map.index<2>) capturing {[[OUT:%[^ :,]+]] = [[OT]] : !mogg.tensor, [[BCASTARG:%[^ :,]+]] = [[BCAST]] : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg.tensor.load [[BCASTARG]]{{\[}}[[IDX]]] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.tensor.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @broadcastSameRank(%input: !mo.tensor<[1, 64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// Rank-expanding: in has one fewer dim than out (numpy semantics). The
// generated mogg.tensor.create.broadcast is identical in shape to the
// rank-preserving case above -- $outShape is the only shape this op carries;
// the input's own (smaller) rank is inferred later, purely from its Mojo
// argument type at emission time.

// CHECK-LABEL: mo.graph @broadcastRankExpanding
// CHECK: mogg.tensor.create.broadcast(%{{.*}}) {{.*}}outShape = #mosh<ape[8, 64]>{{.*}} : !mogg.tensor
mo.graph @broadcastRankExpanding(%input: !mo.tensor<[64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}
