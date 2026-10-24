// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// Static_BroadcastToOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @static_broadcast_to<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>) -> !mo.tensor<[2, [[N]], 3], f32> {
mo.graph @static_broadcast_to<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[2, N, 3], f32> {
  // CHECK: [[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[[[N]], 3], f32> -> !mo.tensor<[2, [[N]], 3], f32>
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[2, N, 3], f32>
  mo.output %res : !mo.tensor<[2, N, 3], f32>
}

// CHECK: mo.graph @static_broadcast_to_reshape<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[5, [[N]]], f32>) -> !mo.tensor<[1, [[N]], 5], f32> {
mo.graph @static_broadcast_to_reshape<N>(%from: !mo.tensor<[5, N], f32>) -> !mo.tensor<[1, N, 5], f32> {
  // CHECK: [[RES:.*]] = mo.static.reshape(%[[FROM]]) : !mo.tensor<[5, [[N]]], f32> -> !mo.tensor<[1, [[N]], 5], f32>
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[5, N], f32> -> !mo.tensor<[1, N, 5], f32>
  mo.output %res : !mo.tensor<[1, N, 5], f32>
}

// CHECK: mo.graph @static_broadcast_to_fold_same_static_rank<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>) -> !mo.tensor<[[[N]], 3], f32> {
mo.graph @static_broadcast_to_fold_same_static_rank<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[N, 3], f32> {
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[N, 3], f32>
  // CHECK: mo.output %[[FROM]] : !mo.tensor<[[[N]], 3], f32>
  mo.output %res : !mo.tensor<[N, 3], f32>
}

// CHECK: mo.graph @static_broadcast_no_reshape<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 1, 0], f32>) -> !mo.tensor<[[[N]], 10, 0], f32> {
mo.graph @static_broadcast_no_reshape<N>(%from: !mo.tensor<[N, 1, 0], f32>) -> !mo.tensor<[N, 10, 0], f32> {
  // CHECK: [[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[[[N]], 1, 0], f32> -> !mo.tensor<[[[N]], 10, 0], f32>
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[N, 1, 0], f32> -> !mo.tensor<[N, 10, 0], f32>
  mo.output %res : !mo.tensor<[N, 10, 0], f32>
}

// CHECK: mo.graph @static_broadcast_fold_consecutive_broadcast(%[[FROM:.*]]: !mo.tensor<[3, 1], f32>) -> !mo.tensor<[5, 3, 5], f32>
mo.graph @static_broadcast_fold_consecutive_broadcast(%from: !mo.tensor<[3, 1], f32>) -> !mo.tensor<[5, 3, 5], f32> {
  // CHECK-NEXT: %[[RES2:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[3, 1], f32> -> !mo.tensor<[5, 3, 5], f32>
  %res1 = mo.static.broadcast_to(%from) : !mo.tensor<[3, 1], f32> -> !mo.tensor<[1, 3, 5], f32>
  %res2 = mo.static.broadcast_to(%res1) : !mo.tensor<[1, 3, 5], f32> -> !mo.tensor<[5, 3, 5], f32>
  mo.output %res2 : !mo.tensor<[5, 3, 5], f32>
}

// CHECK-LABEL: mo.graph @static_broadcast_no_fold_differing_metadata
// CHECK-SAME:  <[[D0:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], 3], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]], 3], f32, {layout = #mo.layout<NHWC>}>
mo.graph @static_broadcast_no_fold_differing_metadata<D0>(
  %input: !mo.tensor<[D0, 3], f32>
) -> !mo.tensor<[D0, 3], f32, {layout = #mo.layout<NHWC>}> {
  // Such pattern could show up from this sequence of passes
  // 1. InferLayouts
  // 2. SymbolicizeFallbackShapeFunctions
  // 3. PropagateShapes

  // CHECK-NEXT: %[[RESULT:.*]] = mo.rebind(%[[INPUT]])
  // CHECK-SAME: : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[D0, 3], f32, {layout = #mo.layout<NHWC>}>
  %result = mo.static.broadcast_to(%input) : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[D0, 3], f32, {layout = #mo.layout<NHWC>}>

  // CHECK-NEXT: mo.output %[[RESULT]] : !mo.tensor<[D0, 3], f32, {layout = #mo.layout<NHWC>}>
  mo.output %result: !mo.tensor<[D0, 3], f32, {layout = #mo.layout<NHWC>}>
}

//===----------------------------------------------------------------------===//
// Static_ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @static_reshape<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>)
mo.graph @static_reshape<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[1, 3, N], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[FROM]]) : !mo.tensor<[[[N]], 3], f32> -> !mo.tensor<[1, 3, [[N]]], f32>
  %res = mo.static.reshape(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[1, 3, N], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1, 3, [[N]]], f32>
  mo.output %res : !mo.tensor<[1, 3, N], f32>
}

// CHECK: mo.graph @static_reshape_handle0<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>)
mo.graph @static_reshape_handle0<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[1, 3, 0], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[FROM]]) : !mo.tensor<[[[N]], 3], f32> -> !mo.tensor<[1, 3, 0], f32>
  %res = mo.static.reshape(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[1, 3, 0], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1, 3, 0], f32>
  mo.output %res : !mo.tensor<[1, 3, 0], f32>
}

// CHECK: mo.graph @static_reshape_fold_same_static_rank<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>)
mo.graph @static_reshape_fold_same_static_rank<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[N, 3], f32> {
  %res = mo.static.reshape(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[N, 3], f32>
  // CHECK-NEXT: mo.output %[[FROM]] : !mo.tensor<[N, 3], f32>
  mo.output %res : !mo.tensor<[N, 3], f32>
}

// CHECK: mo.graph @static_reshape_fold_static_out_shape_attr() -> !mo.tensor<[2, 2], si32> {
mo.graph @static_reshape_fold_static_out_shape_attr() -> !mo.tensor<[2, 2], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %from = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %res = mo.static.reshape(%from) : !mo.tensor<[4], si32> -> !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}

// CHECK: mo.graph @static_reshape_fold_to_rebind_static<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[2, [[B]]], f32> {
mo.graph @static_reshape_fold_to_rebind_static<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[2, B], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[B]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, B], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, B], f32>
  mo.output %0 : !mo.tensor<[2, B], f32>
}

// CHECK: mo.graph @static_reshape_fold_to_rebind_param<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], [[A]]], f32> {
mo.graph @static_reshape_fold_to_rebind_param<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[A, A], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[A]], [[A]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[A, A], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[A]], [[A]]], f32>
  mo.output %0 : !mo.tensor<[A, A], f32>
}

// CHECK: mo.graph @static_reshape_not_fold_to_rebind<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[B]], [[A]]], f32> {
mo.graph @static_reshape_not_fold_to_rebind<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[B, A], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[B]], [[A]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[B, A], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[B]], [[A]]], f32>
  mo.output %0 : !mo.tensor<[B, A], f32>
}

// CHECK: mo.graph @static_reshape_index_const<[[A:.*]]>() -> !mo.tensor<[1], si32> {
mo.graph @static_reshape_index_const<A>() -> !mo.tensor<[1], si32> {
  // CHECK-NEXT: %[[C:.*]] = mosh.param.to_value: !mosh.ape = <[[[A]]]>
  // CHECK-NEXT: %[[RES:.*]] = mo.shape.to_tensor(%[[C]]) -> !mo.tensor<[1], si32>
  %0 = mosh.param.to_value = <A>
  %1 = mo.index.to_tensor(%0) -> !mo.tensor<[], si32>
  %2 = mo.static.reshape(%1) : !mo.tensor<[], si32> -> !mo.tensor<[1], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1], si32>
  mo.output %2 : !mo.tensor<[1], si32>
}

// CHECK: mo.graph @static_reshape_singleton_const<[[A:.*]]>() -> !mo.tensor<[], si64> {
mo.graph @static_reshape_singleton_const<A>() -> !mo.tensor<[], si64> {
  // CHECK-NEXT: %[[C:.*]] = mosh.param.to_value = <[[A]]>
  // CHECK-NEXT: %[[RES:.*]] = mo.index.to_tensor(%[[C]]) -> !mo.tensor<[], si64>
  %0 = mosh.param.to_value: !mosh.ape = <[A]>
  %1 = mo.shape.to_tensor(%0) -> !mo.tensor<[1], si64>
  %2 = mo.static.reshape(%1) : !mo.tensor<[1], si64> -> !mo.tensor<[], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[], si64>
  mo.output %2 : !mo.tensor<[], si64>
}

// CHECK: mo.graph @static_reshape_preceding_rebind<[[A:.*]], [[B:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], si64>) -> !mo.tensor<[10], si64> {
mo.graph @static_reshape_preceding_rebind<A, B>(%arg0: !mo.tensor<[A, B], si64>) -> !mo.tensor<[10], si64> {
  %1 = mo.rebind(%arg0) : !mo.tensor<[A, B], si64> -> !mo.tensor<[2, B], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], si64> -> !mo.tensor<[10], si64>
  %2 = mo.static.reshape(%1) : !mo.tensor<[2, B], si64> -> !mo.tensor<[10], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[10], si64>
  mo.output %2 : !mo.tensor<[10], si64>
}
