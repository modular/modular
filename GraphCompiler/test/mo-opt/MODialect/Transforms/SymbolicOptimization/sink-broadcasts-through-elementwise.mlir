// RUN: mo-opt %s --mo-symbolic-optimize -cse | FileCheck %s

// CHECK: mo.graph @binary_static_rank<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @binary_static_rank<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>, %[[ARG1:.*]]: !mo.tensor<[[[A]], [[B]]], f32>
  %arg0: !mo.tensor<[A, B], f32>, %arg1: !mo.tensor<[A, B], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.sub(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[[[A]], [[B]]], f32>
  %2 = mo.sub(%0, %1) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %2 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @binary_static_rank_scalars<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @binary_static_rank_scalars<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>, %[[ARG1:.*]]: !mo.tensor<[], f32>
  %arg0: !mo.tensor<[A, B], f32>, %arg1: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[BCAST_ARG1:.*]] = mo.static.broadcast_to(%[[ARG1]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[A]], [[B]]], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.sub(%[[ARG0]], %[[BCAST_ARG1]]) : !mo.tensor<[[[A]], [[B]]], f32>
  %2 = mo.sub(%0, %1) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %2 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @binary_static_rank_change_dtypes<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @binary_static_rank_change_dtypes<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>, %[[ARG1:.*]]: !mo.tensor<[[[A]], [[B]]], si32>
  %arg0: !mo.tensor<[A, B], f32>, %arg1: !mo.tensor<[A, B], si32>
// CHECK-SAME: ) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[A, B], si32> -> !mo.tensor<[2, A, B], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.pow(%[[ARG0]] : !mo.tensor<[[[A]], [[B]]], f32>, %[[ARG1]] : !mo.tensor<[[[A]], [[B]]], si32>) : !mo.tensor<[[[A]], [[B]]], f32>
  %2 = mo.pow(%0: !mo.tensor<[2, A, B], f32>, %1: !mo.tensor<[2, A, B], si32>) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %2 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @binary_matching_intermediate_shape<[[A:.*]] -> [[B:.*]]>(
mo.graph @binary_matching_intermediate_shape<A -> B>(
  // CHECK-SAME: %[[ARG0]]: !mo.tensor<[[[A]]], f32>, %[[ARG1]]: !mo.tensor<[2, [[A]]], f32>
  %arg0: !mo.tensor<[A], f32>, %arg1: !mo.tensor<[2, A], f32>
// CHECK-SAME: ) -> !mo.tensor<[3, 2, [[B]]], f32> {
) -> !mo.tensor<[3, 2, B], f32>{
  // CHECK-DAG: %[[ARG0BCAST:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]]], f32> -> !mo.tensor<[2, [[A]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A], f32> -> !mo.tensor<[3, 2, A], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[2, A], f32> -> !mo.tensor<[3, 2, A], f32>
  // CHECK: %[[RES:.*]] = mo.add(%[[ARG0BCAST]], %[[ARG1]]) : !mo.tensor<[2, [[A]]], f32>
  // CHECK: %[[BCASTRES:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[2, [[A]]], f32> -> !mo.tensor<[3, 2, [[A]]], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[3, 2, A], f32>
  // CHECK: mo.output<[[A]]> %[[BCASTRES]] : !mo.tensor<[3, 2, [[A]]], f32>
  mo.output<A> %2 : !mo.tensor<[3, 2, A], f32>
}

// CHECK: mo.graph @binary_no_better_shape
mo.graph @binary_no_better_shape(
  // CHECK-SAME: %[[ARG0]]: !mo.tensor<[1, 5], f32>, %[[ARG1]]: !mo.tensor<[5, 1], f32>
  %arg0: !mo.tensor<[1, 5], f32>, %arg1: !mo.tensor<[5, 1], f32>
// CHECK-SAME: ) -> !mo.tensor<[5, 5], f32> {
) -> !mo.tensor<[5, 5], f32>{
  // CHECK: %[[BCAST0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[1, 5], f32> -> !mo.tensor<[5, 5], f32>
  // CHECK: %[[BCAST1:.*]] = mo.static.broadcast_to(%[[ARG1]]) : !mo.tensor<[5, 1], f32> -> !mo.tensor<[5, 5], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 5], f32> -> !mo.tensor<[5, 5], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[5, 1], f32> -> !mo.tensor<[5, 5], f32>
  // CHECK: %[[RES:.*]] = mo.add(%[[BCAST0]], %[[BCAST1]]) : !mo.tensor<[5, 5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5, 5], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[5, 5], f32>
  mo.output %2 : !mo.tensor<[5, 5], f32>
}

// CHECK: mo.graph @unary_static_rank_change_dtype<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @unary_static_rank_change_dtype<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[2, [[C]], [[D]]], si32> {
    %arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[2, C, D], si32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.cast(%[[ARG0]]) : (!mo.tensor<[A, B], f32>) -> !mo.tensor<[A, B], si32>
  %1 = mo.cast(%0) : (!mo.tensor<[2, A, B], f32>) -> !mo.tensor<[2, A, B], si32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], si32> -> !mo.tensor<[2, [[A]], [[B]]], si32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], si32>
  mo.output<A, B> %1 : !mo.tensor<[2, A, B], si32>
}

// CHECK: mo.graph @unary_static_rank<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @unary_static_rank<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
    %arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32>
  %1 = mo.exp(%0) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %1 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @unary_elemwise_scalars
mo.graph @unary_elemwise_scalars(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[10], f32>
    %arg0: !mo.tensor<[], f32>) -> !mo.tensor<[10], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[10], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[], f32>
  %1 = mo.exp(%0) : !mo.tensor<[10], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[], f32> -> !mo.tensor<[10], f32>
  // CHECK-NEXT: mo.output %[[RESBCAST]] : !mo.tensor<[10], f32>
  mo.output %1 : !mo.tensor<[10], f32>
}

// CHECK: mo.graph @multiple_operands_uses_elemwise
mo.graph @multiple_operands_uses_elemwise(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[10], f32>
    %arg0: !mo.tensor<[], f32>) -> !mo.tensor<[10], f32>{
  // CHECK-DAG: %[[EXP:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[], f32>
  // CHECK-DAG: %[[ERF:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.add(%[[EXP]], %[[ERF]]) : !mo.tensor<[], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[10], f32>
  %1 = mo.exp(%0) : !mo.tensor<[10], f32>
  %2 = mo.erf(%0) : !mo.tensor<[10], f32>
  %3 = mo.add(%1, %2) : !mo.tensor<[10], f32>

  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[], f32> -> !mo.tensor<[10], f32>
  // CHECK-NEXT: mo.output %[[RESBCAST]] : !mo.tensor<[10], f32>
  mo.output %3 : !mo.tensor<[10], f32>
}
