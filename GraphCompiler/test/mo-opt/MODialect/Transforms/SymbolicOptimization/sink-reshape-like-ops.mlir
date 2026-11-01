// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

mo.graph @sink_reshape_through_binary_elemwise(%input: !mo.tensor<[50, 50, 30], f32>, %elemwise: !mo.tensor<[30], f32>) -> !mo.tensor<[2500, 30], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
    %broadcast = mo.static.broadcast_to(%elemwise) : !mo.tensor<[30], f32> -> !mo.tensor<[2500, 30], f32>
    %add = mo.add(%reshape, %broadcast) : !mo.tensor<[2500, 30], f32>
    mo.output %add : !mo.tensor<[2500, 30], f32>
}
// CHECK: mo.graph @sink_reshape_through_binary_elemwise(%[[INPUT:.*]]: !mo.tensor<[50, 50, 30], f32>, %[[ELEMWISE:.*]]: !mo.tensor<[30], f32>) -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: %[[NEW_BROADCAST:.*]] = mo.static.broadcast_to(%[[ELEMWISE]]) : !mo.tensor<[30], f32> -> !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT: %[[NEW_ADD:.*]] = mo.add(%[[INPUT]], %[[NEW_BROADCAST]]) : !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_ADD]]) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[2500, 30], f32>

mo.graph @sink_reshape_through_binary_elemwise_with_device(%input: !mo.tensor<[50, 50, 30], f32, dev:0>, %elemwise: !mo.tensor<[30], f32, dev:0>) -> !mo.tensor<[2500, 30], f32, dev:0>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[50, 50, 30], f32, dev:0> -> !mo.tensor<[2500, 30], f32, dev:0>
    %broadcast = mo.static.broadcast_to(%elemwise) : !mo.tensor<[30], f32, dev:0> -> !mo.tensor<[2500, 30], f32, dev:0>
    %add = mo.add(%reshape, %broadcast) : !mo.tensor<[2500, 30], f32, dev:0>
    mo.output %add : !mo.tensor<[2500, 30], f32, dev:0>
}
// CHECK: mo.graph @sink_reshape_through_binary_elemwise_with_device(%[[INPUT:.*]]: !mo.tensor<[50, 50, 30], f32, dev:0>, %[[ELEMWISE:.*]]: !mo.tensor<[30], f32, dev:0>) -> !mo.tensor<[2500, 30], f32, dev:0>
// CHECK-NEXT: %[[NEW_BROADCAST:.*]] = mo.static.broadcast_to(%[[ELEMWISE]]) : !mo.tensor<[30], f32, dev:0> -> !mo.tensor<[50, 50, 30], f32, dev:0>
// CHECK-NEXT: %[[NEW_ADD:.*]] = mo.add(%[[INPUT]], %[[NEW_BROADCAST]]) : !mo.tensor<[50, 50, 30], f32, dev:0>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_ADD]]) : !mo.tensor<[50, 50, 30], f32, dev:0> -> !mo.tensor<[2500, 30], f32, dev:0>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[2500, 30], f32, dev:0>


mo.graph @sink_reshape_through_binary_elemwise_different_dtypes(%input: !mo.tensor<[50, 50, 30], f32>, %elemwise: !mo.tensor<[30], si32>) -> !mo.tensor<[2500, 30], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
    %broadcast = mo.static.broadcast_to(%elemwise) : !mo.tensor<[30], si32> -> !mo.tensor<[2500, 30], si32>
    %pow = mo.pow(%reshape: !mo.tensor<[2500, 30], f32>, %broadcast: !mo.tensor<[2500, 30], si32>): !mo.tensor<[2500, 30], f32>
    mo.output %pow : !mo.tensor<[2500, 30], f32>
}
// CHECK: mo.graph @sink_reshape_through_binary_elemwise_different_dtypes(%[[INPUT:.*]]: !mo.tensor<[50, 50, 30], f32>, %[[ELEMWISE:.*]]: !mo.tensor<[30], si32>) -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: %[[NEW_BROADCAST:.*]] = mo.static.broadcast_to(%[[ELEMWISE]]) : !mo.tensor<[30], si32> -> !mo.tensor<[50, 50, 30], si32>
// CHECK-NEXT: %[[NEW_POW:.*]] = mo.pow(%[[INPUT]] : !mo.tensor<[50, 50, 30], f32>, %[[NEW_BROADCAST]] : !mo.tensor<[50, 50, 30], si32>) : !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_POW]]) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[2500, 30], f32>

// Above tests matching against mo.add(%a, %b). We test against mo.add(%b, %a) too
// It should give the same graph as above though except the operands are flipped (to preserve ordering for ops like mo.div).
mo.graph @sink_reshape_through_binary_elemwise_flipped_operands(%input: !mo.tensor<[50, 50, 30], f32>, %elemwise: !mo.tensor<[30], f32>) -> !mo.tensor<[2500, 30], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
    %broadcast = mo.static.broadcast_to(%elemwise) : !mo.tensor<[30], f32> -> !mo.tensor<[2500, 30], f32>
    %add = mo.add(%broadcast, %reshape) : !mo.tensor<[2500, 30], f32>
    mo.output %add : !mo.tensor<[2500, 30], f32>
}
// CHECK: mo.graph @sink_reshape_through_binary_elemwise_flipped_operands(%[[INPUT:.*]]: !mo.tensor<[50, 50, 30], f32>, %[[ELEMWISE:.*]]: !mo.tensor<[30], f32>) -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: %[[NEW_BROADCAST:.*]] = mo.static.broadcast_to(%[[ELEMWISE]]) : !mo.tensor<[30], f32> -> !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT: %[[NEW_ADD:.*]] = mo.add(%[[NEW_BROADCAST]], %[[INPUT]]) : !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_ADD]]) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[2500, 30], f32>

// Cannot optimize because the trailing dimensions do not match.
mo.graph @dont_sink_reshape_case(%input: !mo.tensor<[7, 7, 3], f32>, %elemwise: !mo.tensor<[7], f32>) -> !mo.tensor<[7, 3, 7], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[7, 7, 3], f32> -> !mo.tensor<[7, 3, 7], f32>
    %broadcast = mo.static.broadcast_to(%elemwise) : !mo.tensor<[7], f32> -> !mo.tensor<[7, 3, 7], f32>
    %add = mo.add(%broadcast, %reshape) : !mo.tensor<[7, 3, 7], f32>
    mo.output %add : !mo.tensor<[7, 3, 7], f32>
}
// CHECK: mo.graph @dont_sink_reshape_case
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape
// CHECK-NEXT: %[[BROADCAST:.*]] = mo.static.broadcast_to
// CHECK-NEXT: %[[ADD:.*]] = mo.add
// CHECK-NEXT: mo.output %[[ADD]]


mo.graph @sink_double_reshape_through_binary_elemwise(%input1: !mo.tensor<[10, 5], f32>, %input2: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f32>{
    %reshape1 = mo.static.reshape(%input1) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %reshape2 = mo.static.reshape(%input2) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %result = mo.mul(%reshape1, %reshape2) : !mo.tensor<[50], f32>
    mo.output %result : !mo.tensor<[50], f32>
}
// CHECK: mo.graph @sink_double_reshape_through_binary_elemwise(%[[INPUT1:.*]]: !mo.tensor<[10, 5], f32>, %[[INPUT2:.*]]: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f32>
// CHECK-NEXT: %[[MUL:.*]] = mo.mul(%[[INPUT1]], %[[INPUT2]]) : !mo.tensor<[10, 5], f32>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[MUL]]) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
// CHECK-NEXT: mo.output %[[RESHAPE]]

mo.graph @sink_double_reshape_through_binary_elemwise_different_input_dtypes(%input1: !mo.tensor<[10, 5], f32>, %input2: !mo.tensor<[10, 5], si32>) -> !mo.tensor<[50], f32>{
    %reshape1 = mo.static.reshape(%input1) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %reshape2 = mo.static.reshape(%input2) : !mo.tensor<[10, 5], si32> -> !mo.tensor<[50], si32>
    %result = mo.pow(%reshape1 : !mo.tensor<[50], f32>, %reshape2: !mo.tensor<[50], si32>) : !mo.tensor<[50], f32>
    mo.output %result : !mo.tensor<[50], f32>
}
// CHECK: mo.graph @sink_double_reshape_through_binary_elemwise_different_input_dtypes(%[[INPUT1:.*]]: !mo.tensor<[10, 5], f32>, %[[INPUT2:.*]]: !mo.tensor<[10, 5], si32>) -> !mo.tensor<[50], f32>
// CHECK-NEXT: %[[MUL:.*]] = mo.pow(%[[INPUT1]] : !mo.tensor<[10, 5], f32>, %[[INPUT2]] : !mo.tensor<[10, 5], si32>) : !mo.tensor<[10, 5], f32>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[MUL]]) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
// CHECK-NEXT: mo.output %[[RESHAPE]]

// initial shapes are not compatible so cannot sink
mo.graph @dont_sink_double_reshape_through_binary_elemwise(%input1: !mo.tensor<[10, 5], f32>, %input2: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[50], f32>{
    %reshape1 = mo.static.reshape(%input1) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %reshape2 = mo.static.reshape(%input2) : !mo.tensor<[5, 10], f32> -> !mo.tensor<[50], f32>
    %result = mo.mul(%reshape1, %reshape2) : !mo.tensor<[50], f32>
    mo.output %result : !mo.tensor<[50], f32>
}
// CHECK: mo.graph @dont_sink_double_reshape_through_binary_elemwise(%[[INPUT1:.*]]: !mo.tensor<[10, 5], f32>, %[[INPUT2:.*]]: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[50], f32>
// CHECK-NEXT: %[[RESHAPE1:.*]] = mo.static.reshape
// CHECK-NEXT: %[[RESHAPE2:.*]] = mo.static.reshape
// CHECK-NEXT: %[[RESULT:.*]] = mo.mul
// CHECK-NEXT: mo.output %[[RESULT]]

// Above tests matching against mo.add(%a, %b). We test against mo.add(%b, %a) too
// It should give the same graph as above though.
mo.graph @sink_reshape_through_unary_elemwise(%input: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %result = mo.erf(%reshape) : !mo.tensor<[50], f32>
    mo.output %result : !mo.tensor<[50], f32>
}
// CHECK: mo.graph @sink_reshape_through_unary_elemwise(%[[INPUT:.*]]: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f32>
// CHECK-NEXT: %[[NEW_ERF:.*]] = mo.erf(%[[INPUT]]) : !mo.tensor<[10, 5], f32>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_ERF]]) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[50], f32>

mo.graph @sink_reshape_through_unary_elemwise_different_output_dtype(%input: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f64>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[10, 5], f32> -> !mo.tensor<[50], f32>
    %result = mo.cast(%reshape) : (!mo.tensor<[50], f32>) -> !mo.tensor<[50], f64>
    mo.output %result : !mo.tensor<[50], f64>
}
// CHECK: mo.graph @sink_reshape_through_unary_elemwise_different_output_dtype(%[[INPUT:.*]]: !mo.tensor<[10, 5], f32>) -> !mo.tensor<[50], f64>
// CHECK-NEXT: %[[NEW_CAST:.*]] = mo.cast(%[[INPUT]]) : (!mo.tensor<[10, 5], f32>) -> !mo.tensor<[10, 5], f64>
// CHECK-NEXT: %[[NEW_RESHAPE:.*]] = mo.static.reshape(%[[NEW_CAST]]) : !mo.tensor<[10, 5], f64> -> !mo.tensor<[50], f64>
// CHECK-NEXT: mo.output %[[NEW_RESHAPE]] : !mo.tensor<[50], f64>

// See https://github.com/modularml/modular/issues/17518
// This was the initial motivation behind these passes.
// This should take the initial static reshape done before any elemwise
// and put it at the very bottom! Suggest using --view-op-graph flag
// routing std-error to a .dot file and visualizing with GraphViz.
mo.graph @sink_consecutive_elemwise_with_parameters<D8431, P49533, P49878>(
    %input: !mo.tensor<[D8431, 3072], f32>,
    %elemwise1: !mo.tensor<[3072], f32>,
    %elemwise2: !mo.tensor<[], f32>,
    %elemwise3: !mo.tensor<[], f32>,
    %elemwise4: !mo.tensor<[], f32>
) -> !mo.tensor<[P49878, P49533, 3072], f32>{
    kgen.param.declare D8421: si64 = <D8431>
    kgen.param.declare P39675: si64 = <mul_no_wrap(D8431, 3072)>
    kgen.param.declare P39676: si64 = <mul_no_wrap(P49533, P49878, 3072)>
    %280 = mo.static.reshape(%input) : !mo.tensor<[D8431, 3072], f32> -> !mo.tensor<[P49878, P49533, 3072], f32>
    %281 = mo.static.broadcast_to(%elemwise1) : !mo.tensor<[3072], f32> -> !mo.tensor<[P49878, P49533, 3072], f32>
    %282 = mo.add(%280, %281) : !mo.tensor<[P49878, P49533, 3072], f32>
    %283 = mo.static.broadcast_to(%elemwise2) : !mo.tensor<[], f32> -> !mo.tensor<[P49878, P49533, 3072], f32>
    %284 = mo.mul(%282, %283) : !mo.tensor<[P49878, P49533, 3072], f32>
    %285 = mo.static.broadcast_to(%elemwise3) : !mo.tensor<[], f32> -> !mo.tensor<[P49878, P49533, 3072], f32>
    %286 = mo.mul(%282, %285) : !mo.tensor<[P49878, P49533, 3072], f32>
    %287 = mo.erf(%286) : !mo.tensor<[P49878, P49533, 3072], f32>
    %288 = mo.static.broadcast_to(%elemwise4) : !mo.tensor<[], f32> -> !mo.tensor<[P49878, P49533, 3072], f32>
    %289 = mo.add(%287, %288) : !mo.tensor<[P49878, P49533, 3072], f32>
    kgen.param.declare P49316: i1 = <to_builtin(:scalar<bool> eq(:si64 P49878, 1))>
    kgen.param.declare P49317: i1 = <to_builtin(:scalar<bool> eq(:si64 P49878, 1))>
    kgen.param.declare P49318: i1 = <or(P49316, P49317)>
    kgen.param.declare P49325: i1 = <to_builtin(:scalar<bool> eq(:si64 P49533, 1))>
    kgen.param.declare P49326: i1 = <to_builtin(:scalar<bool> eq(:si64 P49533, 1))>
    kgen.param.declare P49327: i1 = <or(P49325, P49326)>
    %290 = mo.mul(%284, %289) : !mo.tensor<[P49878, P49533, 3072], f32>
    mo.output %290 : !mo.tensor<[P49878, P49533, 3072], f32>
}
// Make sure we have all the elemwise ops, and reshape is at the very end
// CHECK: mo.graph @sink_consecutive_elemwise_with_parameters
// CHECK-DAG: %[[ADD1:.*]] = mo.add
// CHECK-DAG: %[[MUL1:.*]] = mo.mul
// CHECK-DAG: %[[MUL2:.*]] = mo.mul
// CHECK-DAG: %[[ERF:.*]] = mo.erf
// CHECK-DAG: %[[ADD2:.*]] = mo.add
// CHECK-DAG: %[[MUL3:.*]] = mo.mul
// CHECK-NEXT: %[[RESHAPE_AT_END:.*]] = mo.static.reshape(%[[MUL3]])
// CHECK-NEXT: mo.output %[[RESHAPE_AT_END]]

mo.graph @sink_unary_reshape_that_squeezes_final_dim(
    %input: !mo.tensor<[2, 1], f32>
) -> !mo.tensor<[2], f32> {
    %reshaped_input = mo.static.reshape(%input) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
    %exp = mo.exp(%reshaped_input) : !mo.tensor<[2], f32>
    mo.output %exp : !mo.tensor<[2], f32>
}
// A reshape that drops a trailing 1 sinks like any other.
// CHECK: mo.graph @sink_unary_reshape_that_squeezes_final_dim(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 1], f32>
// CHECK-SAME: ) -> !mo.tensor<[2], f32>
//
// CHECK-NEXT: %[[EXP:.*]] = mo.exp(%[[INPUT]]) : !mo.tensor<[2, 1], f32>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[EXP]]) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
// CHECK-NEXT: mo.output %[[RESHAPE]] : !mo.tensor<[2], f32>

mo.graph @sink_binary_reshape_that_squeezes_final_dim(
    %lhs: !mo.tensor<[2, 1], f32>,
    %rhs: !mo.tensor<[2, 1], f32>
) -> !mo.tensor<[2], f32> {
    %reshaped_lhs = mo.static.reshape(%lhs) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
    %reshaped_rhs = mo.static.reshape(%rhs) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
    %add = mo.add(%reshaped_lhs, %reshaped_rhs) : !mo.tensor<[2], f32>
    mo.output %add : !mo.tensor<[2], f32>
}
// CHECK: mo.graph @sink_binary_reshape_that_squeezes_final_dim(
// CHECK-SAME: %[[LHS:.*]]: !mo.tensor<[2, 1], f32>,
// CHECK-SAME: %[[RHS:.*]]: !mo.tensor<[2, 1], f32>
// CHECK-SAME: ) -> !mo.tensor<[2], f32>
//
// CHECK-NEXT: %[[ADD:.*]] = mo.add(%[[LHS]], %[[RHS]]) : !mo.tensor<[2, 1], f32>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[ADD]]) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
// CHECK-NEXT: mo.output %[[RESHAPE]] : !mo.tensor<[2], f32>

mo.graph @sink_binary_bcast_reshape_that_squeezes_final_dim(
    %lhs: !mo.tensor<[2, 1], f32>,
    %rhs: !mo.tensor<[], f32>
) -> !mo.tensor<[2], f32> {
    %reshaped_lhs = mo.static.reshape(%lhs) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
    %bcast_rhs = mo.static.broadcast_to(%rhs) : !mo.tensor<[], f32> -> !mo.tensor<[2], f32>
    %add = mo.add(%reshaped_lhs, %bcast_rhs) : !mo.tensor<[2], f32>
    mo.output %add : !mo.tensor<[2], f32>
}
// CHECK: mo.graph @sink_binary_bcast_reshape_that_squeezes_final_dim(
// CHECK-SAME: %[[LHS:.*]]: !mo.tensor<[2, 1], f32>,
// CHECK-SAME: %[[RHS:.*]]: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[2], f32>
//
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[RHS]]) : !mo.tensor<[], f32> -> !mo.tensor<[2, 1], f32>
// CHECK-NEXT: %[[ADD:.*]] = mo.add(%[[LHS]], %[[BCAST]]) : !mo.tensor<[2, 1], f32>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[ADD]]) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
// CHECK-NEXT: mo.output %[[RESHAPE]] : !mo.tensor<[2], f32>


mo.graph @sink_rebind_through_unary_elemwise<D0, D1>(%in: !mo.tensor<[D0, 2, 4], f32>) -> !mo.tensor<[D1, 2, 4], f32> {
  %0 = mo.rebind(%in) : !mo.tensor<[D0, 2, 4], f32> -> !mo.tensor<[D1, 2, 4], f32>
  %1 = mo.exp(%0) : !mo.tensor<[D1, 2, 4], f32>
  mo.output %1 : !mo.tensor<[D1, 2, 4], f32>
}

// CHECK:       mo.graph @sink_rebind_through_unary_elemwise<[[D0:.*]], [[D1:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[D0]], 2, 4], f32>) -> !mo.tensor<[[[D1]], 2, 4], f32>
// CHECK-NEXT:    %[[EXP:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[[[D0]], 2, 4], f32>
// CHECK-NEXT:    %[[REBIND:.*]] = mo.rebind(%[[EXP]]) : !mo.tensor<[[[D0]], 2, 4], f32> -> !mo.tensor<[[[D1]], 2, 4], f32>
// CHECK-NEXT:    mo.output %[[REBIND]] : !mo.tensor<[[[D1]], 2, 4], f32>

mo.graph @sink_reshape_through_softmax<D0>(%in: !mo.tensor<[D0, 2, 3, 4], f32>) -> !mo.tensor<[D0, 6, 4], f32> {
  %0 = mo.static.reshape(%in) : !mo.tensor<[D0, 2, 3, 4], f32> -> !mo.tensor<[D0, 6, 4], f32>
  %1 = mo.reduce.softmax(%0) [axis = 2] : (!mo.tensor<[D0, 6, 4], f32>) -> !mo.tensor<[D0, 6, 4], f32>
  mo.output %1 : !mo.tensor<[D0, 6, 4], f32>
}

// CHECK:       mo.graph @sink_reshape_through_softmax<[[D0:.*]]>(%[[IN:.*]]: !mo.tensor<[[[D0]], 2, 3, 4], f32>) -> !mo.tensor<[[[D0]], 6, 4], f32>
// CHECK-NEXT:    %[[SOFTMAX:.*]] = mo.reduce.softmax(%[[IN]]) [axis = 3] : (!mo.tensor<[[[D0]], 2, 3, 4], f32>) -> !mo.tensor<[[[D0]], 2, 3, 4], f32>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%[[SOFTMAX]]) : !mo.tensor<[[[D0]], 2, 3, 4], f32> -> !mo.tensor<[[[D0]], 6, 4], f32>
// CHECK-NEXT:    mo.output %[[RESHAPE]] : !mo.tensor<[[[D0]], 6, 4], f32>

mo.graph @dont_sink_reshape_through_softmax<D0>(%in: !mo.tensor<[D0, 2, 3, 4], f32>) -> !mo.tensor<[D0, 2, 12], f32> {
  %0 = mo.static.reshape(%in) : !mo.tensor<[D0, 2, 3, 4], f32> -> !mo.tensor<[D0, 2, 12], f32>
  %1 = mo.reduce.softmax(%0) [axis = 2] : (!mo.tensor<[D0, 2, 12], f32>) -> !mo.tensor<[D0, 2, 12], f32>
  mo.output %1 : !mo.tensor<[D0, 2, 12], f32>
}

// CHECK:       mo.graph @dont_sink_reshape_through_softmax<[[D0:.*]]>(%[[IN:.*]]: !mo.tensor<[[[D0]], 2, 3, 4], f32>) -> !mo.tensor<[[[D0]], 2, 12], f32>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%[[IN]]) : !mo.tensor<[[[D0]], 2, 3, 4], f32> -> !mo.tensor<[[[D0]], 2, 12], f32>
// CHECK-NEXT:    %[[SOFTMAX:.*]] = mo.reduce.softmax(%[[RESHAPE]]) [axis = 2] : (!mo.tensor<[[[D0]], 2, 12], f32>) -> !mo.tensor<[[[D0]], 2, 12], f32>
// CHECK-NEXT:    mo.output %[[SOFTMAX]] : !mo.tensor<[[[D0]], 2, 12], f32>

mo.graph @sink_reshape_through_softmax_bad_axis<D0>(%in: !mo.tensor<[D0, 2, 3, 4], f32>) -> !mo.tensor<[D0, 6, 4], f32> {
  %0 = mo.static.reshape(%in) : !mo.tensor<[D0, 2, 3, 4], f32> -> !mo.tensor<[D0, 6, 4], f32>
  %1 = mo.reduce.softmax(%0) [axis = 1] : (!mo.tensor<[D0, 6, 4], f32>) -> !mo.tensor<[D0, 6, 4], f32>
  mo.output %1 : !mo.tensor<[D0, 6, 4], f32>
}

// CHECK:       mo.graph @sink_reshape_through_softmax_bad_axis<[[D0:.*]]>(%[[IN:.*]]: !mo.tensor<[[[D0]], 2, 3, 4], f32>) -> !mo.tensor<[[[D0]], 6, 4], f32>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%[[IN]]) : !mo.tensor<[D0, 2, 3, 4], f32> -> !mo.tensor<[D0, 6, 4], f32>
// CHECK-NEXT:    %[[SOFTMAX:.*]] = mo.reduce.softmax(%[[RESHAPE]]) [axis = 1] : (!mo.tensor<[[[D0]], 6, 4], f32>) -> !mo.tensor<[[[D0]], 6, 4], f32>
// CHECK-NEXT:    mo.output %[[SOFTMAX]] : !mo.tensor<[[[D0]], 6, 4], f32>

mo.graph @sink_reshape_through_softmax_scalar(%in: !mo.tensor<[], f32>) -> !mo.tensor<[1], f32> {
  %0 = mo.static.reshape(%in) : !mo.tensor<[], f32> -> !mo.tensor<[1], f32>
  %1 = mo.reduce.softmax(%0) [axis = 0] : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %1 : !mo.tensor<[1], f32>
}

// CHECK:       mo.graph @sink_reshape_through_softmax_scalar(%[[IN:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[1], f32>
// CHECK-NEXT:    %[[SOFTMAX:.*]] = mo.reduce.softmax(%[[IN]]) [axis = 0] : (!mo.tensor<[], f32>) -> !mo.tensor<[], f32>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%[[SOFTMAX]]) : !mo.tensor<[], f32> -> !mo.tensor<[1], f32>
// CHECK-NEXT:    mo.output %[[RESHAPE]] : !mo.tensor<[1], f32>

// The reshaped operand is the exponent, an si32: the sunk pow keeps its own
// f32 result.
mo.graph @sink_reshape_of_exponent_through_pow(%base: !mo.tensor<[30], f32>, %exp: !mo.tensor<[50, 50, 30], si32>) -> !mo.tensor<[2500, 30], f32> {
    %reshape = mo.static.reshape(%exp) : !mo.tensor<[50, 50, 30], si32> -> !mo.tensor<[2500, 30], si32>
    %broadcast = mo.static.broadcast_to(%base) : !mo.tensor<[30], f32> -> !mo.tensor<[2500, 30], f32>
    %pow = mo.pow(%broadcast: !mo.tensor<[2500, 30], f32>, %reshape: !mo.tensor<[2500, 30], si32>): !mo.tensor<[2500, 30], f32>
    mo.output %pow : !mo.tensor<[2500, 30], f32>
}
// CHECK-LABEL: mo.graph @sink_reshape_of_exponent_through_pow
// CHECK:       %[[POW:.*]] = mo.pow({{.*}} : !mo.tensor<[50, 50, 30], f32>, {{.*}} : !mo.tensor<[50, 50, 30], si32>) : !mo.tensor<[50, 50, 30], f32>
// CHECK-NEXT:  %[[RESHAPE:.*]] = mo.static.reshape(%[[POW]]) : !mo.tensor<[50, 50, 30], f32> -> !mo.tensor<[2500, 30], f32>

// A reshape from [1] to rank 0 drops the last axis, so it stays above the
// softmax.
mo.graph @dont_sink_reshape_to_rank0_through_softmax(%in: !mo.tensor<[1], f32>) -> !mo.tensor<[], f32> {
  %0 = mo.static.reshape(%in) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
  %1 = mo.reduce.softmax(%0) [axis = 0] : (!mo.tensor<[], f32>) -> !mo.tensor<[], f32>
  mo.output %1 : !mo.tensor<[], f32>
}
// CHECK-LABEL: mo.graph @dont_sink_reshape_to_rank0_through_softmax
// CHECK:       mo.static.reshape(%{{.*}}) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
// CHECK-NEXT:  mo.reduce.softmax
