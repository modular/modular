// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, mo.graph(map-fuse-kernels), map-outline-structured-kernel-bodies, mo.graph(canonicalize), map-to-mogg)' | FileCheck %s

// Full pipeline from a real `mo.constant` fed to a broadcast: `LoadOp`'s
// canonicalizer turns the per-element read of the one-element constant into a
// `map.constant.scalar`, which lowers to `mogg.constant.scalar` inside the
// `mogg.call.foreach` region. The constant buffer is not a kernel operand at
// all -- no `!mogg.pointer` input, no `mogg.tensor.create` for it.

// The elementwise-operand shape: the literal takes the place of one of the
// add's two loads, so the call comes out `f32, f32 -> f32` exactly as it
// would with two real loads.

// CHECK-LABEL: mo.graph @addConstant
// CHECK:       mogg.routine ({{.*}} : !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]])
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]])
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8, 64]> : !mosh.ape>
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg.tensor.create([[IP]], [[ISH]]) {{.*}}io = Input{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8, 64]> : !mosh.ape>
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) {{.*}}io = Output{{.*}} : !mogg.tensor
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8, 64]> : !mosh.ape>
// CHECK-NEXT:    mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) ([[IDX:%[^ :,]+]] : !map.index<2>) capturing {[[IN:%[^ :,]+]] = [[IT]] : !mogg.tensor, [[OUT:%[^ :,]+]] = [[OT]] : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:      [[CST:%[^ ]+]] = mogg.constant.scalar 2.000000e+00 : f32 : !mogg.simd<f32>
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg.tensor.load [[IN]]{{\[}}[[IDX]]] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
// CHECK-NEXT:      [[SUM:%[^ ]+]] = mogg.call.elementwise @builtin_kernels.elementwise.Add([[VAL]], [[CST]]) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.tensor.store [[OUT]]{{\[}}[[IDX]]] = [[SUM]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
mo.graph @addConstant(%x: !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[8, 64], f32>
  %r = mo.add(%x, %b) : !mo.tensor<[8, 64], f32>
  mo.output %r : !mo.tensor<[8, 64], f32>
}

// An unfused broadcast of a constant is a whole kernel whose only job was the
// read: it lowers to a kernel with no input pointer at all, storing the
// literal.

// CHECK-LABEL: mo.graph @broadcastConstant
// CHECK:       mogg.routine () -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  outputs ([[OP2:%[^ :,]+]])
// CHECK:         mogg.call.foreach {{.*}} ([[IDX2:%[^ :,]+]] : !map.index<2>) capturing {[[OUT2:%[^ :,]+]] = {{[^ ]+}} : !mogg.tensor}
// CHECK-NEXT:      [[CST2:%[^ ]+]] = mogg.constant.scalar 5.000000e-01 : f32 : !mogg.simd<f32>
// CHECK-NEXT:      mogg.tensor.store [[OUT2]]{{\[}}[[IDX2]]] = [[CST2]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
mo.graph @broadcastConstant() -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant <#M.dense_array<5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %b : !mo.tensor<[8, 64], f32>
}
