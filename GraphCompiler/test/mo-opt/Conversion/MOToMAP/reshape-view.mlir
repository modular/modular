// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.static.reshape lowers to a standalone (unfused)
// map.iter.elementwise over the output shape, with a map.index.reshape op
// computing the input index -- not map.iter.opaque, and not the "view
// kernels are not yet supported" error.

// Rank-reducing case: flatten [4, 6] into [24].

// CHECK-LABEL: mo.graph @reshapeFlatten
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT:%.*]] = map.iter.elementwise<[24]>
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[4, 6], f32>)
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[24], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<1>)  -> !mo.tensor<[24], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.reshape<#map.index.reshape.spec<[4, 6] -> [24]>> ([[IDX]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[4, 6], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[24], f32>
mo.graph @reshapeFlatten(%input: !mo.tensor<[4, 6], f32>) -> !mo.tensor<[24], f32> {
  %res = mo.static.reshape(%input) : !mo.tensor<[4, 6], f32> -> !mo.tensor<[24], f32>
  mo.output %res : !mo.tensor<[24], f32>
}

// Rank-expanding case: unflatten [24] into [4, 6].

// CHECK-LABEL: mo.graph @reshapeExpand
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT2:%.*]] = map.iter.elementwise<[4, 6]>
// CHECK:         inputs  ([[INPUT2:%.*]]: !mo.tensor<[24], f32>)
// CHECK:         outputs ([[OUTARG2:%.*]]: !mo.tensor<[4, 6], f32>)
// CHECK:         index   ([[IDX2:%.*]]: !map.index<2>)  -> !mo.tensor<[4, 6], f32> {
// CHECK:         [[IDX2_:%.*]] = map.index.reshape<#map.index.reshape.spec<[24] -> [4, 6]>> ([[IDX2]])
// CHECK:         [[VAL2:%.*]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[24], f32>
// CHECK:         map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 6], f32>
mo.graph @reshapeExpand(%input: !mo.tensor<[24], f32>) -> !mo.tensor<[4, 6], f32> {
  %res = mo.static.reshape(%input) : !mo.tensor<[24], f32> -> !mo.tensor<[4, 6], f32>
  mo.output %res : !mo.tensor<[4, 6], f32>
}

// Rank-preserving case: [2, 6] to [3, 4] -- same rank, different shape.

// CHECK-LABEL: mo.graph @reshapeSameRank
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[3, 4]>
// CHECK:         map.index.reshape<#map.index.reshape.spec<[2, 6] -> [3, 4]>>
mo.graph @reshapeSameRank(%input: !mo.tensor<[2, 6], f32>) -> !mo.tensor<[3, 4], f32> {
  %res = mo.static.reshape(%input) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// Symbolic in/out shape dims (N): proving the lowering works when the static
// shapes aren't fully concrete either.

// CHECK-LABEL: mo.graph @reshapeSymbolicDims
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[N, 2, 3]>
// CHECK:         map.index.reshape<#map.index.reshape.spec<[N, 6] -> [N, 2, 3]>>
mo.graph @reshapeSymbolicDims<N>(%input: !mo.tensor<[N, 6], f32>) -> !mo.tensor<[N, 2, 3], f32> {
  %res = mo.static.reshape(%input) : !mo.tensor<[N, 6], f32> -> !mo.tensor<[N, 2, 3], f32>
  mo.output %res : !mo.tensor<[N, 2, 3], f32>
}
