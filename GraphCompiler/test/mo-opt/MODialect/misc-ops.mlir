// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @broadcast_shape(%[[ARG0:.*]]: !mo.tensor<[3], si64>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
mo.graph @broadcast_shape(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// CHECK: mo.graph @broadcast_shape_i32(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
mo.graph @broadcast_shape_i32(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[3], si32>
  mo.output %res : !mo.tensor<[3], si32>
}

// CHECK: mo.graph @broadcast_shape_partially_ranked<[[D0:.*]]>(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[[[D0]]], si32>) -> !mo.tensor<[4], si32>
mo.graph @broadcast_shape_partially_ranked<D0>(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[D0], si32>) -> !mo.tensor<[4], si32> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D0]]], si32>) -> !mo.tensor<[4], si32>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[D0], si32>) -> !mo.tensor<[4], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @broadcast_same_rank
// CHECK: (%[[FROM:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_same_rank(%from: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %to  = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[1, 3], f32> -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK-LABEL: mo.graph @broadcast_diff_rank
// CHECK: (%[[FROM:.*]]: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_diff_rank(%from: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32> {
  %to  = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @chain_forwarding
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mo.output [[ch0]] : !mo.chain
mo.graph @chain_forwarding() -> !mo.chain
{
  %ch0 = mo.chain.create()
  %ch1 = mo.chain.create(%ch0) : !mo.chain

  mo.output %ch1: !mo.chain
}

// CHECK-LABEL: @duplicate_removal
// CHECK: ([[ch0:%.*]]: !mo.chain, [[ch1:%.*]]: !mo.chain)
// CHECK: [[ch2:%.*]] = mo.chain.create([[ch0]], [[ch1]]) : !mo.chain, !mo.chain
// CHECK: mo.output [[ch2]] : !mo.chain
mo.graph @duplicate_removal(%ch0: !mo.chain, %ch1: !mo.chain) -> !mo.chain
{
  %ch2 = mo.chain.create(%ch0, %ch0, %ch1) : !mo.chain, !mo.chain, !mo.chain
  mo.output %ch2: !mo.chain
}

// CHECK: @dedupe_and_forwarding
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mo.output [[ch0]] : !mo.chain
mo.graph @dedupe_and_forwarding() -> !mo.chain
{
  %ch0 = mo.chain.create()
  %ch1 = mo.chain.create(%ch0, %ch0) : !mo.chain, !mo.chain

  mo.output %ch1: !mo.chain
}

// CHECK-LABEL: @chain_flattening
// CHECK: (%[[ARG0:.*]]: !mo.chain, %[[ARG1:.*]]: !mo.chain, %[[ARG2:.*]]: !mo.chain, %[[ARG3:.*]]: !mo.chain)
mo.graph @chain_flattening(
  %arg0: !mo.chain, %arg1: !mo.chain, %arg2: !mo.chain, %arg3: !mo.chain
) -> !mo.chain {
  %x = mo.chain.create(%arg0, %arg1) : !mo.chain, !mo.chain
  %y = mo.chain.create(%arg2, %arg3) : !mo.chain, !mo.chain

  // CHECK: %[[RES:.*]] = mo.chain.create(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
  %z = mo.chain.create(%x, %y) : !mo.chain, !mo.chain

  // CHECK: mo.output %[[RES]]
  mo.output %z: !mo.chain
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @concat(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: mo.concat[1] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[1](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// CHECK: mo.graph @concat_canonicalize<{{.*}}>(
mo.graph @concat_canonicalize<D0>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, {{.*}}], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 0], f32>, %[[AX:.*]]: !mo.tensor<[], si64>
  %arg0: !mo.tensor<[2, D0], f32>, %arg1: !mo.tensor<[2, 0], f32>, %ax: !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, {{.*}}], f32> {
) -> !mo.tensor<[2, D0], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.concat[1] (%[[ARG0]], %[[ARG0]]) : (
  // CHECK-SAME:   !mo.tensor<[2, {{.*}}], f32>, !mo.tensor<[2, {{.*}}], f32>) -> !mo.tensor<[2, {{.*}}], f32>
  %0 = mo.concat[1](%arg0, %arg1, %arg0) : (
    !mo.tensor<[2, D0], f32>, !mo.tensor<[2, 0], f32>, !mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, {{.*}}], f32>
  mo.output %0 : !mo.tensor<[2, D0], f32>
}

// CHECK: mo.graph @concat_canonicalize_to_rebind<[[N:.*]], [[K:.*]] -> [[M:.*]]>(
mo.graph @concat_canonicalize_to_rebind<N, K -> M>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, [[N]]], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 0], f32>, %[[AX:.*]]: !mo.tensor<[], si64>
  %arg0: !mo.tensor<[2, N], f32>, %arg1: !mo.tensor<[2, 0], f32>, %ax: !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, [[M]]], f32> {
) -> !mo.tensor<[2, M], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[2, N], f32> -> !mo.tensor<[2, K], f32>
  %0 = mo.concat[1](%arg1, %arg0, %arg1) : (
    !mo.tensor<[2, 0], f32>, !mo.tensor<[2, N], f32>, !mo.tensor<[2, 0], f32>) -> !mo.tensor<[2, K], f32>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[2, [[K]]], f32>
  mo.output<K> %0 : !mo.tensor<[2, K], f32>
}

// CHECK: mo.graph @concat_canonicalize_single_input
mo.graph @concat_canonicalize_single_input<D0>(
  %arg0: !mo.tensor<[2, D0], f32>,  %ax: !mo.tensor<[], si64>
) -> !mo.tensor<[2, D0], f32> {

  // CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[2, {{.*}}], f32>
  %0 = mo.concat[1](%arg0) : (!mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32>
  mo.output %0 : !mo.tensor<[2, D0], f32>
}

// CHECK-LABEL: mo.graph @concat_do_not_canonicalize_param_decls
mo.graph @concat_do_not_canonicalize_param_decls<A -> B>(
  %arg0: !mo.tensor<[2, A], f32>,  %ax: !mo.tensor<[], si64>
) -> !mo.tensor<[2, B], f32> {

  // CHECK: mo.concat<() -> O1>
  %0 = mo.concat<() -> O1>[1](%arg0) : (!mo.tensor<[2, A], f32>) -> !mo.tensor<[2, O1], f32>
  mo.output<O1> %0 : !mo.tensor<[2, O1], f32>
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @split_static<[[D0:[^,>]*]], [[D1:[^,>]*]]>(%[[ARG0:.*]]: !mo.tensor<[6], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, !mo.tensor<{{\[}}[[D1]]{{\]}}, si32>) {
mo.graph @split_static<D0, D1>(%arg0: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>) {
  // CHECK: %{{.*}} = mo.split(%[[ARG0]], %[[ARG1]]) [axis = 0] : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (!mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, !mo.tensor<{{\[}}[[D1]]{{\]}}, si32>)
  %out1, %out2 = mo.split(%arg0, %arg1) [axis = 0] : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (
    !mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>
  )
  mo.output %out1, %out2 : !mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>
}

// CHECK: mo.graph @split_negative_axis<[[D0:[^,>]*]], [[D1:[^,>]*]]>(%[[ARG0:.*]]: !mo.tensor<[6], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, !mo.tensor<{{\[}}[[D1]]{{\]}}, si32>) {
mo.graph @split_negative_axis<D0, D1>(%arg0: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>) {
  // CHECK: %{{.*}} = mo.split(%[[ARG0]], %[[ARG1]]) [axis = -1] : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (!mo.tensor<{{\[}}[[D0]]{{\]}}, si32>, !mo.tensor<{{\[}}[[D1]]{{\]}}, si32>)
  %out1, %out2 = mo.split(%arg0, %arg1) [axis = -1] : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (
    !mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>
  )
  mo.output %out1, %out2 : !mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @gather(%[[INPUT:.*]]: !mo.tensor<[2, 2], f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: mo.gather(%[[INPUT]], %[[INDICES]]) [axis = 0] : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @pad_static_output_shape(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 5], f32> {
mo.graph @pad_static_output_shape(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 5], f32> {
  // CHECK: %[[CONSTANT:.*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  // CHECK: %[[PADDINGS:.*]] = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  // CHECK: mo.pad.constant(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 5], f32>
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 5], f32>
  mo.output %res : !mo.tensor<[3, 5], f32>
}

// CHECK: mo.graph @pad_dynamic_output_shape<[[D0:[^,>]*]], [[D1:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32> {
mo.graph @pad_dynamic_output_shape<D0, D1>(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1], f32> {
  // CHECK: %[[CONSTANT:.*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> :
  %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  // CHECK: mo.pad.constant(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>) -> !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %res : !mo.tensor<[D0, D1], f32>
}

// CHECK: mo.graph @pad_zero_paddings<[[D0:[^,>]*]], [[D1:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<[2, 2], f32>) -> !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>
mo.graph @pad_zero_paddings<D0, D1>(%input: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, D1], f32> {

  // CHECK-NEXT:   %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>
  %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0, D1], f32>

  // CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>
  mo.output %res : !mo.tensor<[D0, D1], f32>
}

// CHECK: mo.graph @pad_zero_paddings_with_output_param<() -> [[D1:.*]], [[D2:.*]]>(%[[INPUT:.*]]: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[[[D1]], [[D2]]], f32>
mo.graph @pad_zero_paddings_with_output_param<() -> D1, D2>(%input: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D1, D2], f32> {

  // CHECK-DAG: kgen.param.declare [[D3:.*]]: scalar<si64> = <2>
  // CHECK-DAG: kgen.param.declare [[D4:.*]]: scalar<si64> = <2>
  // CHECK-NEXT:   %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
  %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %res = mo.pad.constant<() -> D3, D4>(%input, %paddings, %constant) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[D3, D4], f32>

  // CHECK-NEXT:   mo.output<{{.*}}> %[[RES]]
  mo.output<D3, D4> %res : !mo.tensor<[D3, D4], f32>
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @range_float_dyn<[[D0:[^,>]*]]>(%[[LIMIT:.*]]: !mo.tensor<[], f32>)
mo.graph @range_float_dyn<D0>(%limit: !mo.tensor<[], f32>) -> !mo.tensor<[D0], f32> {
  // CHECK-DAG: %[[START:.*]] = mo.constant <#M.dense_array<0.{{0+}}e+00> : tensor<f32>> : !mo.tensor<[], f32>
  // CHECK-DAG: %[[STEP:.*]] = mo.constant <#M.dense_array<1.5{{0+}}e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %start = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %step = mo.constant <#M.dense_array<1.5> : tensor<f32>> : !mo.tensor<[], f32>
  // CHECK: %[[RES:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (
  %res = mo.range(%start, %limit, %step) : (
    // CHECK-SAME: !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<{{\[}}[[D0]]{{\]}}, f32>
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0], f32>
  mo.output %res : !mo.tensor<[D0], f32>
}

// CHECK: mo.graph @range_int_static(%[[START:.*]]: !mo.tensor<[], si32>)
mo.graph @range_int_static(%startInt: !mo.tensor<[], si32>) -> !mo.tensor<[5], si32> {
  // CHECK-DAG: %[[STEP:.*]] = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK-DAG: %[[LIMIT:.*]] = mo.constant <#M.dense_array<11> : tensor<si32>> : !mo.tensor<[], si32>
  %stepInt = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[], si32>
  %limitInt = mo.constant <#M.dense_array<11> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK: %[[RES:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (
  %res = mo.range(%startInt, %limitInt, %stepInt) : (
    // CHECK-SAME: !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5], si32>
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5], si32>
  mo.output %res : !mo.tensor<[5], si32>
}

// CHECK: mo.graph @range_folder() -> (!mo.tensor<[5], ui32>, !mo.tensor<[4], si64>) {
mo.graph @range_folder() -> (!mo.tensor<[5], ui32>, !mo.tensor<[4], si64>) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant <#M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>> : !mo.tensor<[5], ui32>
  %start = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
  %limit = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
  %step = mo.constant <#M.dense_array<2> : tensor<ui32>> : !mo.tensor<[], ui32>
  %0 = mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[5], ui32>
  // CHECK-DAG: %[[VAL1:.*]] = mo.constant <#M.dense_array<2, 1, 0, -1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %startNeg = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
  %limitNeg = mo.constant <#M.dense_array<-2> : tensor<si64>> : !mo.tensor<[], si64>
  %stepNeg = mo.constant <#M.dense_array<-1> : tensor<si64>> : !mo.tensor<[], si64>
  %1 = mo.range(%startNeg, %limitNeg, %stepNeg) : (
    !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[4], si64>
  // CHECK-NEXT: %[[VAL0]], %[[VAL1]] : !mo.tensor<[5], ui32>, !mo.tensor<[4], si64>
  mo.output %0, %1 : !mo.tensor<[5], ui32>, !mo.tensor<[4], si64>
}

// CHECK: mo.graph @range_folder_unknown_output<[[D0:[^,>]*]]>() -> !mo.tensor<{{\[}}[[D0]]{{\]}}, ui32> {
mo.graph @range_folder_unknown_output<D0>() -> !mo.tensor<[D0], ui32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.constant <#M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>> : !mo.tensor<[5], ui32>
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[5], ui32> -> !mo.tensor<{{\[}}[[D0]]{{\]}}, ui32>
  %start = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
  %limit = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
  %step = mo.constant <#M.dense_array<2> : tensor<ui32>> : !mo.tensor<[], ui32>
  %0 = mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[D0], ui32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D0]]{{\]}}, ui32>
  mo.output %0 : !mo.tensor<[D0], ui32>
}

// CHECK: mo.graph @range_folder_parametric_output<() -> [[D0:.*]]>() -> !mo.tensor<[[[D0]]], ui32> {
mo.graph @range_folder_parametric_output<() -> D0>() -> !mo.tensor<[D0], ui32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.constant <#M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>> : !mo.tensor<[5], ui32>
  // CHECK-DAG: kgen.param.declare [[D1:.*]]: scalar<si64> = <5>
  // CHECK-DAG: %[[RES:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[5], ui32> -> !mo.tensor<[[[D1]]], ui32>
  %start = mo.constant <#M.dense_array<1> : tensor<ui32>> : !mo.tensor<[], ui32>
  %limit = mo.constant <#M.dense_array<11> : tensor<ui32>> : !mo.tensor<[], ui32>
  %step = mo.constant <#M.dense_array<2> : tensor<ui32>> : !mo.tensor<[], ui32>
  %0 = mo.range<() -> D1>(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[D1], ui32>
  // CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], ui32>
  mo.output<D1> %0 : !mo.tensor<[D1], ui32>
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @reshape_static_shape(%[[ARG:.*]]: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @reshape_static_shape(%arg: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.static.reshape(%[[ARG]]) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 2], f32>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @reshape_fold_rank_1_to_scalar() -> !mo.tensor<[], si32>
mo.graph @reshape_fold_rank_1_to_scalar() -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant <#M.dense_array<42> : tensor<si32>> : !mo.tensor<[], si32>
  %input = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %shape = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[], si32>
  mo.output %res : !mo.tensor<[], si32>
}

// CHECK: mo.graph @reshape_fold_scalar_to_rank_1() -> !mo.tensor<[1], si32>
mo.graph @reshape_fold_scalar_to_rank_1() -> !mo.tensor<[1], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %input = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %shape = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[1], si32>
  mo.output %res : !mo.tensor<[1], si32>
}

// CHECK: mo.graph @reshape_fold_rank_1_to_scalar_dyn_target_shape(%[[SHAPE:.*]]: !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
mo.graph @reshape_fold_rank_1_to_scalar_dyn_target_shape(%shape: !mo.tensor<[1], si64>) -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant <#M.dense_array<42> : tensor<si32>> : !mo.tensor<[], si32>
  %input = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[], si32>
  mo.output %res : !mo.tensor<[], si32>
}

// CHECK: mo.graph @reshape_fold_scalar_to_rank_1_dyn_target_shape<() -> [[D0:.*]]>(
mo.graph @reshape_fold_scalar_to_rank_1_dyn_target_shape<() -> D0>(
  // CHECK-SAME: %[[SHAPE:.*]]: !mo.tensor<[1], si64>
  %shape: !mo.tensor<[1], si64>
// CHECK-SAME: ) -> !mo.tensor<[D0], si32> {
) -> !mo.tensor<[D0], si32> {
  // CHECK-DAG: %[[INPUT:.*]] = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %input = mo.constant <#M.dense_array<42> : tensor<1xsi32>> : !mo.tensor<[], si32>
  // CHECK-DAG: kgen.param.declare D1: scalar<si64> = <1>
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[1], si32> -> !mo.tensor<[[[D1]]], si32>
  %res = mo.reshape<() -> D1>(%input, %shape) : (!mo.tensor<[], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D1], si32>
  // CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], si32>
  mo.output<D1> %res : !mo.tensor<[D1], si32>
}

// CHECK: mo.graph @reshape_fold_multi_dims() -> !mo.tensor<[2, 2], si32>
mo.graph @reshape_fold_multi_dims() -> !mo.tensor<[2, 2], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %input = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @squeeze_shape
// CHECK-NEXT: %[[RESULT1:.*]] = mo.constant <#M.dense_array<0, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-NEXT: %[[RESULT2:.*]] = mo.squeeze_shape(%[[___:.*]], %[[RESULT1]]) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: mo.output %1 : !mo.tensor<[1], si32>
mo.graph @squeeze_shape(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[1], si32> {
  %indices = mo.constant <#M.dense_array<0, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  mo.output %res : !mo.tensor<[1], si32>
}

// CHECK: mo.graph @squeeze_dynamic_shape<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>
mo.graph @squeeze_dynamic_shape<D0, D1, D2>(%input_shape: !mo.tensor<[D0], si32>, %indices: !mo.tensor<[D1], si32>) -> !mo.tensor<[D2], si32> {
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>) -> !mo.tensor<[D2], si32>
  mo.output %res : !mo.tensor<[D2], si32>
}

// CHECK: mo.graph @squeeze_shape_canonicalize
// CHECK-NEXT: %[[RESULT1:.*]] = mo.constant <#M.dense_array<0, 1, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
mo.graph @squeeze_shape_canonicalize(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[1], si32> {
  // -1 refers to index 3, -3 refers to index 1 (aka <0, 1, 3>)
  %indices = mo.constant <#M.dense_array<-1, -3, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1], si32>
  mo.output %res : !mo.tensor<[1], si32>
}

// Indices whose last pair is already in order still get sorted.
// CHECK: mo.graph @squeeze_shape_canonicalize_unsorted
// CHECK-NEXT: %[[RESULT1:.*]] = mo.constant <#M.dense_array<0, 1, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
mo.graph @squeeze_shape_canonicalize_unsorted(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[1], si32> {
  %indices = mo.constant <#M.dense_array<3, 0, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1], si32>
  mo.output %res : !mo.tensor<[1], si32>
}

// An empty constant index set is valid and removes nothing.
// CHECK: mo.graph @squeeze_shape_empty_indices
mo.graph @squeeze_shape_empty_indices(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %indices = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// UnsqueezeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @unsqueeze_shape
// %[[RESULT1:.*]] = mo.constant <#M.dense_array<0, 2, 6> : tensor<3xsi64>> : !mo.tensor<[3], si64>
// %[[RESULT2:.*]] = mo.unsqueeze_shape(%[[___:.*]], %[[RESULT1]]) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[7], si32>
// mo.output %1 : !mo.tensor<[7], si32>
mo.graph @unsqueeze_shape(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[7], si32> {
  %indices = mo.constant <#M.dense_array<0, 2, 6> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[7], si32>
  mo.output %res : !mo.tensor<[7], si32>
}

// CHECK: mo.graph @unsqueeze_dynamic_shape<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>
mo.graph @unsqueeze_dynamic_shape<D0, D1, D2>(%input_shape: !mo.tensor<[D0], si32>, %indices: !mo.tensor<[D1], si32>) -> !mo.tensor<[D2], si32> {
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>) -> !mo.tensor<[D2], si32>
  mo.output %res : !mo.tensor<[D2], si32>
}

// CHECK: mo.graph @unsqueeze_shape_canonicalize
// %[[RESULT1:.*]] = mo.constant <#M.dense_array<0, 2, 6> : tensor<3xsi64>> : !mo.tensor<[3], si64>
mo.graph @unsqueeze_shape_canonicalize(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[7], si32> {
  // -1 refers to index 6, -7 refers to index 0 (aka <0, 2, 6>)
  %indices = mo.constant <#M.dense_array<-1, -7, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
  mo.output %res : !mo.tensor<[7], si32>
}

// An empty constant index set is valid and inserts nothing.
// CHECK: mo.graph @unsqueeze_shape_empty_indices
mo.graph @unsqueeze_shape_empty_indices(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %indices = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @scatter_nd(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3], f32>,
// CHECK-SAME:   %[[UPDATES:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[1, 2, 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3], f32>
mo.graph @scatter_nd(
  %input: !mo.tensor<[4, 3], f32>,
  %updates: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[1, 2, 1], si32>
) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[RES:.*]] = mo.scatter_nd(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.scatter_nd(%input, %updates, %indices)
  : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

//===----------------------------------------------------------------------===//
// ScatterNdAddOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @scatter_nd.add(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3], f32>,
// CHECK-SAME:   %[[UPDATES:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[1, 2, 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3], f32>
mo.graph @scatter_nd.add(
  %input: !mo.tensor<[4, 3], f32>,
  %updates: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[1, 2, 1], si32>
) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[RES:.*]] = mo.scatter_nd.add(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.scatter_nd.add(%input, %updates, %indices)
  : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

//===----------------------------------------------------------------------===//
// ScatterNdMaxOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @scatter_nd.max(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3], f32>,
// CHECK-SAME:   %[[UPDATES:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[1, 2, 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3], f32>
mo.graph @scatter_nd.max(
  %input: !mo.tensor<[4, 3], f32>,
  %updates: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[1, 2, 1], si32>
) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[RES:.*]] = mo.scatter_nd.max(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.scatter_nd.max(%input, %updates, %indices)
  : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

//===----------------------------------------------------------------------===//
// ScatterNdMinOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @scatter_nd.min(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3], f32>,
// CHECK-SAME:   %[[UPDATES:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[1, 2, 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3], f32>
mo.graph @scatter_nd.min(
  %input: !mo.tensor<[4, 3], f32>,
  %updates: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[1, 2, 1], si32>
) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[RES:.*]] = mo.scatter_nd.min(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.scatter_nd.min(%input, %updates, %indices)
  : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

//===----------------------------------------------------------------------===//
// ScatterNdSubOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @scatter_nd.mul(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3], f32>,
// CHECK-SAME:   %[[UPDATES:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[1, 2, 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3], f32>
mo.graph @scatter_nd.mul(
  %input: !mo.tensor<[4, 3], f32>,
  %updates: !mo.tensor<[1, 2, 3], f32>,
  %indices: !mo.tensor<[1, 2, 1], si32>
) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[RES:.*]] = mo.scatter_nd.mul(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.scatter_nd.mul(%input, %updates, %indices)
  : (!mo.tensor<[4, 3], f32>, !mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 1], si32>) -> !mo.tensor<[4, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @shape_of_static_rank(%[[ARG:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
mo.graph @shape_of_static_rank(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64> {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<1, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2], si64>
  mo.output %res : !mo.tensor<[2], si64>
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @slice_static_rank(%[[INPUT:.*]]: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[8, 5], f32> {
mo.graph @slice_static_rank(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[8, 5], f32> {
  // CHECK-NEXT: %[[BEGIN:.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[END:.*]] = mo.constant <#M.dense_array<-1, 10> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %stop = mo.constant <#M.dense_array<-1, 10> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  // CHECK-NEXT: %[[STRIDES:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[BEGIN]], %[[END]], %[[STRIDES]]) : (
  %res = mo.slice(%input, %start, %stop, %step) : (
    // CHECK-SAME: !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[8, 5], f32>
  ) -> !mo.tensor<[8, 5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// CHECK: mo.graph @slice_dyn_rank<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]], [[D3:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, f32> {
mo.graph @slice_dyn_rank<D0, D1, D2, D3>(%input: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32> {
  // CHECK-NEXT: %[[BEGIN:.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[END:.*]] = mo.constant <#M.dense_array<-1, 10> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %stop = mo.constant <#M.dense_array<-1, 10> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  // CHECK-NEXT: %[[STRIDES:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[BEGIN]], %[[END]], %[[STRIDES]]) : (
  %res = mo.slice(%input, %start, %stop, %step) : (
    // CHECK-SAME: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, f32>
  ) -> !mo.tensor<[D2, D3], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, f32>
  mo.output %res : !mo.tensor<[D2, D3], f32>
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @transpose(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
mo.graph @transpose(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  // CHECK: %[[PERM:.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: mo.transpose(%[[INPUT]], %[[PERM]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// CHECK: mo.graph @transpose_identity_fold_no_decl<[[D0:[^,>]*]], [[D1:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32> {
mo.graph @transpose_identity_fold_no_decl<D0, D1>(%input: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %perm = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>
  mo.output %res : !mo.tensor<[D0, D1], f32>
}

// CHECK: mo.graph @transpose_identity_fold_with_decl<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @transpose_identity_fold_with_decl<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 42], f32>
  %input: !mo.tensor<[D0, 42], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]]], f32> {
) -> !mo.tensor<[D1, D2], f32> {
  %perm = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: kgen.param.declare [[D3:.*]]: scalar<si64> = <[[D0]]>
  // CHECK-DAG: kgen.param.declare [[D4:.*]]: scalar<si64> = <42>
  // CHECK-DAG: %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[[[D0]], 42], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
  %res = mo.transpose<() -> D3, D4>(%input, %perm) : (!mo.tensor<[D0, 42], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
  // CHECK-NEXT: mo.output<[[D3]], [[D4]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]]], f32>
  mo.output<D3, D4> %res : !mo.tensor<[D3, D4], f32>
}

// CHECK: mo.graph @transpose_consecutive_fold<[[D0:.*]] -> [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @transpose_consecutive_fold<D0 -> D1, D2, D3>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 10, 20], f32>
  %input: !mo.tensor<[D0, 10, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> {
) -> !mo.tensor<[D1, D2, D3], f32> {
  // CHECK-NEXT: %[[NEW_PERM:.*]] = mo.constant <#M.dense_array<2, 1, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[PERM1:.*]] = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %perm1 = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %perm2 = mo.constant <#M.dense_array<1, 2, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RES1:.*]] = mo.transpose<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]]>(%[[INPUT]], %[[PERM1]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>
  %res1 = mo.transpose<() -> D4, D5, D6>(%input, %perm1) : (!mo.tensor<[D0, 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D4, D5, D6], f32>
  // CHECK-NEXT: %[[RES2:.*]] = mo.transpose<() -> [[D7:.*]], [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[NEW_PERM]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
  %res2 = mo.transpose<() -> D7, D8, D9>(%res1, %perm2) : (!mo.tensor<[D4, D5, D6], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D7, D8, D9], f32>
  // CHECK-NEXT: mo.output<[[D7]], [[D8]], [[D9]]> %[[RES2]] : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
  mo.output<D7, D8, D9> %res2 : !mo.tensor<[D7, D8, D9], f32>
}

// CHECK: mo.graph @transpose_non_const_perm_no_fold(
mo.graph @transpose_non_const_perm_no_fold(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2], f32>,
  %input: !mo.tensor<[2, 2], f32>,
  // CHECK-SAME: %[[PERM:.*]]: !mo.tensor<[2], si64>
  %perm: !mo.tensor<[2], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, 2], f32> {
) -> !mo.tensor<[2, 2], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.transpose(%[[INPUT]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @transpose_simplify_1_dims_to_reshape(
mo.graph @transpose_simplify_1_dims_to_reshape(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[5, 1, 1, 8, 1, 1, 10], f32>
  %input: !mo.tensor<[5, 1, 1, 8, 1, 1, 10], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32> {
) -> !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32> {
  // CHECK-NEXT: %[[FOLDED:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[5, 1, 1, 8, 1, 1, 10], f32>
  // CHECK-SAME: -> !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32>
  %perm = mo.constant <#M.dense_array<2, 1, 0, 3, 6, 4, 5> : tensor<7xsi64>> : !mo.tensor<[7], si64>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[5, 1, 1, 8, 1, 1, 10], f32>, !mo.tensor<[7], si64>) -> !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32>
  // CHECK-NEXT: mo.output %[[FOLDED]] : !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32>
  mo.output %res : !mo.tensor<[1, 1, 5, 8, 10, 1, 1], f32>
}

// tricky case, but still changing relative order of non-1 dims, so no simplify
// CHECK: mo.graph @transpose_no_simplify_to_reshape_if_ooo(
mo.graph @transpose_no_simplify_to_reshape_if_ooo(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[8, 1, 8], f32>
  %input: !mo.tensor<[8, 1, 8], f32>
// CHECK-SAME: ) -> !mo.tensor<[8, 1, 8], f32> {
) -> !mo.tensor<[8, 1, 8], f32> {
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant <#M.dense_array<2, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %perm = mo.constant <#M.dense_array<2, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.transpose(%[[INPUT]], %[[PERM]]) : (!mo.tensor<[8, 1, 8], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[8, 1, 8], f32>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[8, 1, 8], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[8, 1, 8], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[8, 1, 8], f32>
  mo.output %res : !mo.tensor<[8, 1, 8], f32>
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @arg_nonzero_1d<[[D0:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<[7], f32>) -> !mo.tensor<{{\[}}[[D0]], 1{{\]}}, si64>
mo.graph @arg_nonzero_1d<D0>(%arg: !mo.tensor<[7], f32>) -> !mo.tensor<[D0, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero(%[[ARG]]) : (!mo.tensor<[7], f32>) -> !mo.tensor<{{\[}}[[D0]], 1{{\]}}, si64>
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[7], f32>) -> !mo.tensor<[D0, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D0]], 1{{\]}}, si64>
  mo.output %res : !mo.tensor<[D0, 1], si64>
}

// CHECK: mo.graph @arg_nonzero_2d<[[D0:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<[7, 7], f32>) -> !mo.tensor<{{\[}}[[D0]], 2{{\]}}, si64>
mo.graph @arg_nonzero_2d<D0>(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[D0, 2], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero(%[[ARG]]) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<{{\[}}[[D0]], 2{{\]}}, si64>
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[D0, 2], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D0]], 2{{\]}}, si64>
  mo.output %res : !mo.tensor<[D0, 2], si64>
}

// CHECK: mo.graph @arg_nonzero_2d_dynamic_shape<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 2{{\]}}, si64>
mo.graph @arg_nonzero_2d_dynamic_shape<D0, D1, D2>(%arg: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 2], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero(%[[ARG]]) : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 2{{\]}}, si64>
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 2], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], 2{{\]}}, si64>
  mo.output %res : !mo.tensor<[D2, 2], si64>
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @arg_max_2d_inner_axis(%[[ARG:.*]]: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
mo.graph @arg_max_2d_inner_axis(%arg: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_max(%[[ARG]]) [axis = 1] : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[7, 1], si64>
  mo.output %res : !mo.tensor<[7, 1], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_dynamic_shape<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
mo.graph @arg_max_2d_inner_axis_dynamic_shape<D0, D1, D2>(%arg: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_max(%[[ARG]]) [axis = 1] : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
  mo.output %res : !mo.tensor<[D2, 1], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_dynamic_reduced_dim<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]], [[D3:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
mo.graph @arg_max_2d_inner_axis_dynamic_reduced_dim<D0, D1, D2, D3>(%arg: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_max(%[[ARG]]) [axis = 1] : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
  mo.output %res : !mo.tensor<[D2, D3], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_int32(%[[ARG:.*]]: !mo.tensor<[7, 3], si32>) -> !mo.tensor<[7, 1], si32>
mo.graph @arg_max_2d_inner_axis_int32(%arg: !mo.tensor<[7, 3], si32>) -> !mo.tensor<[7, 1], si32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_max(%[[ARG]]) [axis = 1] : (!mo.tensor<[7, 3], si32>) -> !mo.tensor<[7, 1], si32>
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[7, 3], si32>) -> !mo.tensor<[7, 1], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[7, 1], si32>
  mo.output %res : !mo.tensor<[7, 1], si32>
}

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @arg_min_2d_inner_axis(%[[ARG:.*]]: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
mo.graph @arg_min_2d_inner_axis(%arg: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_min(%[[ARG]]) [axis = 1] : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
  %res = mo.reduce.arg_min(%arg) [axis = 1] : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[7, 1], si64>
  mo.output %res : !mo.tensor<[7, 1], si64>
}

// CHECK: mo.graph @arg_min_2d_inner_axis_dynamic_shape<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
mo.graph @arg_min_2d_inner_axis_dynamic_shape<D0, D1, D2>(%arg: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_min(%[[ARG]]) [axis = 1] : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
  %res = mo.reduce.arg_min(%arg) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], 1{{\]}}, si64>
  mo.output %res : !mo.tensor<[D2, 1], si64>
}

// CHECK: mo.graph @arg_min_2d_inner_axis_dynamic_reduced_dim<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]], [[D3:[^,>]*]]>(%[[ARG:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
mo.graph @arg_min_2d_inner_axis_dynamic_reduced_dim<D0, D1, D2, D3>(%arg: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.arg_min(%[[ARG]]) [axis = 1] : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
  %res = mo.reduce.arg_min(%arg) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, si64>
  mo.output %res : !mo.tensor<[D2, D3], si64>
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @tile(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32> {
mo.graph @tile(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  // CHECK: %[[REPEATS:.*]] = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: mo.tile(%[[INPUT]], %[[REPEATS]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32>
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

// CHECK: mo.graph @tile_dynamic<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]], [[D3:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, f32> {
mo.graph @tile_dynamic<D0, D1, D2, D3>(%input: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32> {
  // CHECK: %[[REPEATS:.*]] = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: mo.tile(%[[INPUT]], %[[REPEATS]]) : (!mo.tensor<{{\[}}[[D0]], [[D1]]{{\]}}, f32>, !mo.tensor<[2], si64>) -> !mo.tensor<{{\[}}[[D2]], [[D3]]{{\]}}, f32>
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output %res : !mo.tensor<[D2, D3], f32>
}

// A symbolic input dim blocks the broadcast rewrite: the lowered form would
// need the host-resident repeats to size a device-side shape tensor.
// CHECK: mo.graph @tile_broadcast<[[D0:[^,>]*]], [[D1:[^,>]*]], [[D2:[^,>]*]]>(%[[INPUT:.*]]: !mo.tensor<[1, [[D0]]], f32>) -> !mo.tensor<{{\[}}[[D1]], [[D2]]{{\]}}, f32> {
mo.graph @tile_broadcast<D0, D1, D2>(%input: !mo.tensor<[1, D0], f32>) -> !mo.tensor<[D1, D2], f32> {
  // CHECK: %[[REPEATS:.*]] = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: mo.tile(%[[INPUT]], %[[REPEATS]]) : (!mo.tensor<[1, [[D0]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<{{\[}}[[D1]], [[D2]]{{\]}}, f32>
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[1, D0], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D1, D2], f32>
  // CHECK: mo.output %[[RES:.*]] : !mo.tensor<{{\[}}[[D1]], [[D2]]{{\]}}, f32>
  mo.output %res : !mo.tensor<[D1, D2], f32>
}

// Statically known broadcastable tile: input dim 1 with repeat 2 lowers to a
// static broadcast, keeping the host repeats out of the device shape ops.
// CHECK: mo.graph @tile_broadcast_static(%[[INPUT:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @tile_broadcast_static(%input: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK-NOT: mo.tile
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: %[[BCST:.*]] = mo.static.broadcast_to(%[[INPUT]]) : !mo.tensor<[1, 3], f32> -> !mo.tensor<[2, 3], f32>
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[BCST]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// A device-resident input stays on its device: only the repeats are host.
// CHECK: mo.graph @tile_gpu(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[4, 3], f32, gpu:0> {
mo.graph @tile_gpu(%input: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[4, 3], f32, gpu:0> {
  // CHECK: %[[REPEATS:.*]] = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.tile(%[[INPUT]], %[[REPEATS]]) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32, gpu:0>
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32, gpu:0>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4, 3], f32, gpu:0>
  mo.output %res : !mo.tensor<[4, 3], f32, gpu:0>
}
