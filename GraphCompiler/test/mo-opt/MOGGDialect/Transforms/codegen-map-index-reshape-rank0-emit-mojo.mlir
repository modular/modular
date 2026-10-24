// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A reduce to a one-element vector, reshaped to a rank-0 scalar and fused into
// the reduce's epilogue as a store-index transform. `reduce.mul` still lowers to
// a kernel with an epilogue. Every emitter represents a
// rank-0 shape as rank-1 extent-1, so the rank handed to mogg_index_reshape has
// to say 1 as well -- a `to_rank` of 0 against a rank-1 `to_static_shape` cannot
// bind, and the store it feeds is a rank-1 tile tensor either way.

// CHECK: mogg_index_reshape[to_rank = 1, from_static_shape = [[SHAPE:.*]], to_static_shape = [[SHAPE]]]
// CHECK-NOT: mogg_index_reshape[to_rank = 0

mo.graph @reduce_reshape_to_scalar() -> (!mo.tensor<[2], f32>) {
  %x = mo.constant <#M.dense_array<1.0, 1.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %y = mo.constant <#M.dense_array<4.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %res = mo.while (%x as %xi: !mo.tensor<[2], f32>) {
    %prod0 = mo.reduce.mul(%xi) [axis = 0] : (!mo.tensor<[2], f32>) -> !mo.tensor<[1], f32>
    %prod1 = mo.static.reshape(%prod0) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
    %cond = mo.greater(%y, %prod1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
    mo.while.condition(%cond) %xi : !mo.tensor<[2], f32>
  } do {
    %new_x = mo.add(%xi, %xi) : !mo.tensor<[2], f32>
    mo.yield %new_x : !mo.tensor<[2], f32>
  }
  mo.output %res : !mo.tensor<[2], f32>
}
