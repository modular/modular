// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A reshape on a kernel's INPUT side fuses into its prologue, so
// `map.index.reshape` has to be legal there -- every sibling index op already
// is. Nothing else was needed: the emitter and the index transform both
// handle a prologue parent, only the op's parent list did not admit one.

// A rowwise reduce has no separate prologue: its input-side ops, the reshape
// among them, fuse into the fold body.
// CHECK-LABEL: mo.graph @reshape_into_reduce_prologue
// CHECK: map.iter.rowwise
// CHECK: map.iter.reduce
// CHECK-NEXT: map.index.reshape
mo.graph @reshape_into_reduce_prologue(%a: !mo.tensor<[1, 4], f32>) -> (!mo.tensor<[1], f32>) {
  %r = mo.static.reshape(%a) : !mo.tensor<[1, 4], f32> -> !mo.tensor<[4], f32>
  %m_sum = mo.reduce.add(%r) : (!mo.tensor<[4], f32>) -> !mo.tensor<[1], f32>
  %m_len = mo.constant <#M.dense_array<4.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %m_len_b = mo.static.broadcast_to(%m_len) : !mo.tensor<[], f32> -> !mo.tensor<[1], f32>
  %m = mo.div(%m_sum, %m_len_b) : !mo.tensor<[1], f32>
  mo.output %m : !mo.tensor<[1], f32>
}
