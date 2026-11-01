// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize)), map-to-mogg)' | FileCheck %s

// Full pipeline from a real mo.transpose op through MOToMAP, outlining, view
// motion, and MAPToMOGG: map.index.transpose.stride lowers to
// mogg.tensor.create.transpose, which carries the view's own output shape and
// the compile-time-known part of the perm (`staticPermutations`) -- the input's
// shape/rank is read off the input tensor operand at Mojo-emission time, not
// duplicated onto this op. perm rides in as a runtime tensor operand too
// (reconstructed as mogg.tensor.create like any other kernel input); a
// constant perm never reads it at runtime, but it stays live because the view
// primitive takes it. The `canonicalize` right after LoopInvariantViewMotion
// drops the closure's now-dead original input capture (unused once the
// hoisted transpose view replaces the load) via
// `DropDeadCapturingClosureCaptures`.

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK:       [[KERNEL:%[^ ]+]] = mogg.routine {{(dispatch\([a-z]+\) )?}}({{.*}} : !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]], [[PP:%[^ :,]+]])
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]])
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[2, 3]> : !mosh.ape>
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg.tensor.create([[IP]], [[ISH]]) {{.*}}io = Input{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[PSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[2]> : !mosh.ape>
// CHECK-NEXT:    [[PT:%[^ ]+]] = mogg.tensor.create([[PP]], [[PSH]]) {{.*}}io = Input{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[3, 2]> : !mosh.ape>
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) {{.*}}io = Output{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[TRANS:%[^ ]+]] = mogg.tensor.create.transpose([[IT]], [[PT]]) {{.*}}outShape = #mosh<ape[3, 2]>{{.*}}staticPermutations = array<i64: 1, 0>{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[3, 2]> : !mosh.ape>
// CHECK-NEXT:    mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) ([[IDX:%[^ :,]+]] : !map.index<2>) capturing {[[OUT:%[^ :,]+]] = [[OT]] : !mogg.tensor, [[TRANSARG:%[^ :,]+]] = [[TRANS]] : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg.tensor.load [[TRANSARG]]{{\[}}[[IDX]]] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.tensor.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// 3D permutation: the generated mogg.tensor.create.transpose carries the
// full non-trivial perm; the input's own (differently-ordered) shape is
// inferred later, purely from its Mojo argument type at emission time.

// CHECK-LABEL: mo.graph @transpose3D
// CHECK: mogg.tensor.create.transpose(%{{.*}}, %{{.*}}) {{.*}}outShape = #mosh<ape[4, 2, 3]>{{.*}}staticPermutations = array<i64: 2, 0, 1>{{.*}} : !mogg.tensor
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[4, 2, 3], f32> {
  %perm = mo.constant <#M.dense_array<2, 0, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[4, 2, 3], f32>
  mo.output %res : !mo.tensor<[4, 2, 3], f32>
}
