// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.static.broadcast_to lowers to a standalone (unfused)
// map.iter.elementwise over the output shape, with a map.index.broadcast op
// computing the input index -- not map.iter.opaque, and not the "view
// kernels are not yet supported" error.

// Rank-preserving case: in-shape dim 0 (size 1) broadcasts to 8.

// CHECK-LABEL: mo.graph @broadcastSameRank
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT:%.*]] = map.iter.elementwise<[8, 64]>
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[1, 64], f32>)
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[8, 64], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[8, 64], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>> ([[IDX]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[1, 64], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[8, 64], f32>
mo.graph @broadcastSameRank(%input: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// Rank-expanding case (numpy semantics): in has one fewer dim than out, so
// the leading output dim is entirely replicated and dropped from the
// projected index.

// CHECK-LABEL: mo.graph @broadcastRankExpanding
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT2:%.*]] = map.iter.elementwise<[8, 64]>
// CHECK:         inputs  ([[INPUT2:%.*]]: !mo.tensor<[64], f32>)
// CHECK:         outputs ([[OUTARG2:%.*]]: !mo.tensor<[8, 64], f32>)
// CHECK:         index   ([[IDX2:%.*]]: !map.index<2>) -> !mo.tensor<[8, 64], f32> {
// CHECK:         [[IDX2_:%.*]] = map.index.broadcast<#map.index.broadcast.spec<[64] -> [8, 64]>> ([[IDX2]])
// CHECK:         [[VAL2:%.*]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[64], f32>
// CHECK:         map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[8, 64], f32>
mo.graph @broadcastRankExpanding(%input: !mo.tensor<[64], f32>) -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// Symbolic in/out shape dims (N, M): proving the lowering works when the
// static shapes aren't fully concrete either.

// CHECK-LABEL: mo.graph @broadcastSymbolicDims
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[M, 64]>
// CHECK:         map.index.broadcast<#map.index.broadcast.spec<[N, 64] -> [M, 64]>>
mo.graph @broadcastSymbolicDims<N, M>(%input: !mo.tensor<[N, 64], f32>) -> !mo.tensor<[M, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[N, 64], f32> -> !mo.tensor<[M, 64], f32>
  mo.output %res : !mo.tensor<[M, 64], f32>
}
