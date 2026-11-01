// RUN: mo-opt %s --mo-symbolic-optimize -canonicalize -cse | FileCheck %s --enable-var-scope

// Standard case, run reduction on axis not affected by the bcast
mo.graph @reduce_sinking_success<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[P1, P2, P3, 1, B], f32>, !mo.tensor<[P1, P2, P3, A, 1], f32>) {

  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[P1, P2, P3, 1, B], f32>
  %reduce_axis2_sum = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[P1, P2, P3, A, 1], f32>
  %reduce_axis2_len_idx = mosh.param.to_value = <B>
  %reduce_axis2_len_u = mo.index.to_tensor(%reduce_axis2_len_idx) -> !mo.tensor<[], ui64>
  %reduce_axis2_len = mo.cast(%reduce_axis2_len_u) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
  %reduce_axis2_len_b = mo.static.broadcast_to(%reduce_axis2_len) : !mo.tensor<[], f32> -> !mo.tensor<[P1, P2, P3, A, 1], f32>
  %reduce_axis2 = mo.div(%reduce_axis2_sum, %reduce_axis2_len_b) : !mo.tensor<[P1, P2, P3, A, 1], f32>

  mo.output %reduce_axis1, %reduce_axis2: !mo.tensor<[P1, P2, P3, 1, B], f32>, !mo.tensor<[P1, P2, P3, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> (!mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>, !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>) {
// CHECK-NEXT: [[V2:%.*]] = mo.reduce.add([[INPUT]]) : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[1, [[B]]], f32>
// CHECK-NEXT: [[V3:%.*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[1, [[B]]], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>
// CHECK-NEXT: [[V4:%.*]] = mo.reduce.add([[INPUT]]) : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>
// CHECK-NEXT: [[LEN_IDX:%.*]] = mosh.param.to_value = <[[B]]>
// CHECK-NEXT: [[LEN_U:%.*]] = mo.index.to_tensor([[LEN_IDX]]) -> !mo.tensor<[], ui64>
// CHECK-NEXT: [[LEN:%.*]] = mo.cast([[LEN_U]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK-NEXT: [[LEN_B:%.*]] = mo.static.broadcast_to([[LEN]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[A]], 1], f32>
// CHECK-NEXT: [[MEAN:%.*]] = mo.div([[V4]], [[LEN_B]]) : !mo.tensor<[[[A]], 1], f32>
// CHECK-NEXT: [[V5:%.*]] = mo.static.broadcast_to([[MEAN]]) : !mo.tensor<[[[A]], 1], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>
// CHECK-NEXT: mo.output [[V3]], [[V5]] : !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>, !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>

// No sinking occurs because we cannot determine which axis are the result of broadcasting
mo.graph @reduce_sinking_failure_cannot_prove_bcasting<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, P2, P3, 1, P3], f32> {

  // COM: B --> P3 might be a broadcasted axis depending on the runtime values
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, P3], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, P3], f32>) -> !mo.tensor<[P1, P2, P3, 1, P3], f32>

  mo.output %reduce_axis1: !mo.tensor<[P1, P2, P3, 1, P3], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_failure_cannot_prove_bcasting<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>(
// CHECK-SAME: [[VARG0:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[P3]]], f32>
// CHECK-NEXT: [[VBROADCASTED_ARG0:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], [[P3]]], f32>
// CHECK-NEXT: [[VREDUCE_AXIS1:%.*]] = mo.reduce.add([[VBROADCASTED_ARG0]]) : (!mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], [[P3]]], f32>) -> !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[P3]]], f32>
// CHECK-NEXT: mo.output [[VREDUCE_AXIS1]] : !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[P3]]], f32>

// Reduce-max on an axis that was broadcasted has no effect. A mean (sum then
// divide) on such an axis scales the input by the axis length, then divides it
// back out.
mo.graph @reduce_sinking_success_max_mean_on_new_bcasted_dim<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[1, P2, P3, A, B], f32>, !mo.tensor<[P1, 1, P3, A, B], f32>) {

  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.max(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[1, P2, P3, A, B], f32>
  %reduce_axis2_sum = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[P1, 1, P3, A, B], f32>
  %reduce_axis2_len_idx = mosh.param.to_value = <P2>
  %reduce_axis2_len_u = mo.index.to_tensor(%reduce_axis2_len_idx) -> !mo.tensor<[], ui64>
  %reduce_axis2_len = mo.cast(%reduce_axis2_len_u) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
  %reduce_axis2_len_b = mo.static.broadcast_to(%reduce_axis2_len) : !mo.tensor<[], f32> -> !mo.tensor<[P1, 1, P3, A, B], f32>
  %reduce_axis2 = mo.div(%reduce_axis2_sum, %reduce_axis2_len_b) : !mo.tensor<[P1, 1, P3, A, B], f32>

  mo.output %reduce_axis1, %reduce_axis2: !mo.tensor<[1, P2, P3, A, B], f32>, !mo.tensor<[P1, 1, P3, A, B], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_max_mean_on_new_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> (!mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>) {
// CHECK-NEXT: [[V0:%.*]] = mo.static.broadcast_to([[INPUT]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: [[LEN_IDX:%.*]] = mosh.param.to_value = <[[P2]]>
// CHECK-NEXT: [[LEN_U:%.*]] = mo.index.to_tensor([[LEN_IDX]]) -> !mo.tensor<[], ui64>
// CHECK-NEXT: [[LEN:%.*]] = mo.cast([[LEN_U]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK-NEXT: [[LEN_B:%.*]] = mo.static.broadcast_to([[LEN]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[SUM:%.*]] = mo.mul([[INPUT]], [[LEN_B]]) : !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[MEAN:%.*]] = mo.div([[SUM]], [[LEN_B]]) : !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[V1:%.*]] = mo.static.broadcast_to([[MEAN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: mo.output [[V0]], [[V1]] : !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>

// Doing 1-broadcasted + reduce_min on the same axis is an identity.
mo.graph @reduce_sinking_success_min_on_1_bcasted_dim<A, B>(
  %arg0: !mo.tensor<[A, 1, B], f32>
) -> !mo.tensor<[A, 1, B], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, 1, B], f32> -> !mo.tensor<[A, 5, B], f32>
  %reduce_axis1 = mo.reduce.min(%broadcasted_arg0) : (!mo.tensor<[A, 5, B], f32>) -> !mo.tensor<[A, 1, B], f32>

  mo.output %reduce_axis1: !mo.tensor<[A, 1, B], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_min_on_1_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], 1, [[B]]], f32>) -> !mo.tensor<[[[A]], 1, [[B]]], f32> {
// CHECK-NEXT: mo.output [[INPUT]] : !mo.tensor<[[[A]], 1, [[B]]], f32>

// Reduce.add on an axis that was broadcasted => input scaled with mo.mul.
mo.graph @reduce_sinking_success_add_on_new_bcasted_dim<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[1, P2, P3, A, B], f32>) {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[1, P2, P3, A, B], f32>
  mo.output %reduce_axis1 : !mo.tensor<[1, P2, P3, A, B], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_add_on_new_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32> {
// CHECK-NEXT: [[INDEX:%.*]] = mosh.param.to_value = <[[P1]]>
// CHECK-NEXT: [[V0:%.*]] = mo.index.to_tensor([[INDEX]]) -> !mo.tensor<[], ui64>
// CHECK-NEXT: [[V1:%.*]] = mo.cast([[V0]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[V3:%.*]] = mo.mul([[INPUT]], [[V2]]) : !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[V4:%.*]] = mo.static.broadcast_to([[V3]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: mo.output [[V4]] : !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>

// Reduce.mul on an axis that was broadcasted => input scaled with mo.pow.
mo.graph @reduce_sinking_success_mul_on_new_bcasted_dim<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[1, P2, P3, A, B], f32>) {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.mul(%broadcasted_arg0) [axis = 0] : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[1, P2, P3, A, B], f32>
  mo.output %reduce_axis1 : !mo.tensor<[1, P2, P3, A, B], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_mul_on_new_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32> {
// CHECK-NEXT: [[INDEX:%.*]] = mosh.param.to_value = <[[P1]]>
// CHECK-NEXT: [[V0:%.*]] = mo.index.to_tensor([[INDEX]]) -> !mo.tensor<[], ui64>
// CHECK-NEXT: [[V1:%.*]] = mo.cast([[V0]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[V3:%.*]] = mo.pow([[INPUT]] : !mo.tensor<[[[A]], [[B]]], f32>, [[V2]] : !mo.tensor<[[[A]], [[B]]], f32>) : !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-NEXT: [[V4:%.*]] = mo.static.broadcast_to([[V3]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: mo.output [[V4]] : !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>

// arg_min / arg_max on an axis that was broadcasted => output is all 0s
mo.graph @reduce_sinking_success_argmin_on_new_bcasted_dim<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[1, P2, P3, A, B], si64>, !mo.tensor<[1, P2, P3, A, B], si64>) {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %argmin_axis1 = mo.reduce.arg_min(%broadcasted_arg0) [axis = 0] : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[1, P2, P3, A, B], si64>
  %argmax_axis1 = mo.reduce.arg_max(%broadcasted_arg0) [axis = 0] : (!mo.tensor<[P1, P2, P3, A, B], f32>) -> !mo.tensor<[1, P2, P3, A, B], si64>
  mo.output %argmin_axis1, %argmax_axis1 : !mo.tensor<[1, P2, P3, A, B], si64>, !mo.tensor<[1, P2, P3, A, B], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_argmin_on_new_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> (!mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], si64>, !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], si64>) {
// CHECK-NEXT: [[V0:%.*]] = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], si64> -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], si64>
// CHECK-NEXT: mo.output [[V1]], [[V1]] : !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], si64>, !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], si64>

//===----------------------------------------------------------------------===//
// Coverage on all reduction operators
//===----------------------------------------------------------------------===//

mo.graph @reduce_sinking_success_mean<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1_sum = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], f32>
  %reduce_axis1_len_idx = mosh.param.to_value = <B>
  %reduce_axis1_len_u = mo.index.to_tensor(%reduce_axis1_len_idx) -> !mo.tensor<[], ui64>
  %reduce_axis1_len = mo.cast(%reduce_axis1_len_u) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
  %reduce_axis1_len_b = mo.static.broadcast_to(%reduce_axis1_len) : !mo.tensor<[], f32> -> !mo.tensor<[P1, A, 1], f32>
  %reduce_axis1 = mo.div(%reduce_axis1_sum, %reduce_axis1_len_b) : !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_mean<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.add
// CHECK-SAME: (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>
// CHECK: mo.div
// CHECK-SAME: !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_mul<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.mul(%broadcasted_arg0) [axis = 2] : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_reduce_mul<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.mul
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_add<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0) : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_reduce_add<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.add
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_min<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.min(%broadcasted_arg0) : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_reduce_min<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.min
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_max<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.max(%broadcasted_arg0) : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_reduce_max<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.max
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_arg_max<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], si64> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.arg_max(%broadcasted_arg0) [axis = 2] : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], si64>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_arg_max<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.arg_max
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], si64>

mo.graph @reduce_sinking_success_arg_min<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], si64> {
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.arg_min(%broadcasted_arg0) [axis = 2] : (!mo.tensor<[P1, A, B], f32>) -> !mo.tensor<[P1, A, 1], si64>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reduce_sinking_success_arg_min<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.arg_min
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], 1], si64>
