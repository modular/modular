// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

// CHECK: mo.graph @gather_to_split<[[D0:.*]]>(
mo.graph @gather_to_split<D0>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[3, [[D0]], 10], f32>,
  %input: !mo.tensor<[3, D0, 10], f32>,
  // CHECK-SAME: %[[DYN_IDX:.*]]: !mo.tensor<[1], si64>
  %dyn_idx: !mo.tensor<[1], si64>
// CHECK-SAME: ) -> !mo.tensor<[1, D0, 10], f32> {
) -> !mo.tensor<[1, D0, 10], f32> {
  %idx0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx1 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx2 = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  // CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK: %[[SPLIT_RES:.*]]:3 = mo.split(%[[INPUT]], %[[SPLIT_SIZES]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[3, [[D0]], 10], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[1, [[D0]], 10], f32>, !mo.tensor<[1, [[D0]], 10], f32>, !mo.tensor<[1, [[D0]], 10], f32>)
  %gather0 = mo.gather(%input, %idx0) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1, D0, 10], f32>
  %gather1 = mo.gather(%input, %idx1) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1, D0, 10], f32>
  %gather2 = mo.gather(%input, %idx2) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1, D0, 10], f32>

  // NOTE extra gather (or other) user of input is okay (left as is)
  // CHECK-NEXT: %[[DYN_GATHER:.*]] = mo.gather(%[[INPUT]], %[[DYN_IDX]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[3, [[D0]], 10], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1, [[D0]], 10], f32>
  %dyn_gather = mo.gather(%input, %dyn_idx) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1, D0, 10], f32>

  // CHECK-NEXT: %[[ADD01:.*]] = mo.add(%[[SPLIT_RES]]#0, %[[SPLIT_RES]]#1) : !mo.tensor<[1, [[D0]], 10], f32>
  %add01 = mo.add(%gather0, %gather1) : !mo.tensor<[1, D0, 10], f32>
  // CHECK-NEXT: %[[ADD012:.*]] = mo.add(%[[ADD01]], %[[SPLIT_RES]]#2) : !mo.tensor<[1, [[D0]], 10], f32>
  %add012 = mo.add(%add01, %gather2) : !mo.tensor<[1, D0, 10], f32>
  // CHECK-NEXT: %[[ADD_ALL:.*]] = mo.add(%[[ADD012]], %[[DYN_GATHER]]) : !mo.tensor<[1, [[D0]], 10], f32>
  %add_all = mo.add(%add012, %dyn_gather) : !mo.tensor<[1, D0, 10], f32>

  // CHECK-NEXT: mo.output %[[ADD_ALL]] : !mo.tensor<[1, [[D0]], 10], f32>
  mo.output %add_all : !mo.tensor<[1, D0, 10], f32>
}

// CHECK: mo.graph @gather_to_split_with_device<[[D0:.*]]>(
mo.graph @gather_to_split_with_device<D0>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[3, [[D0]], 10], f32, dev:0>,
  %input: !mo.tensor<[3, D0, 10], f32, dev:0>,
  // CHECK-SAME: %[[DYN_IDX:.*]]: !mo.tensor<[1], si64>
  %dyn_idx: !mo.tensor<[1], si64>
// CHECK-SAME: ) -> !mo.tensor<[1, D0, 10], f32, dev:0> {
) -> !mo.tensor<[1, D0, 10], f32, dev:0> {
  %idx0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64, dev:0>
  %idx1 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64, dev:0>
  %idx2 = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64, dev:0>

  // CHECK-NEXT: %[[SPLIT_SIZES:.*]] = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[SPLIT_RES:.*]]:3 = mo.split(%[[INPUT]], %[[SPLIT_SIZES]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[3, [[D0]], 10], f32, dev:0>, !mo.tensor<[3], si64>) -> (!mo.tensor<[1, [[D0]], 10], f32, dev:0>, !mo.tensor<[1, [[D0]], 10], f32, dev:0>, !mo.tensor<[1, [[D0]], 10], f32, dev:0>)
  %gather0 = mo.gather(%input, %idx0) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32, dev:0>, !mo.tensor<[1], si64, dev:0>) -> !mo.tensor<[1, D0, 10], f32, dev:0>
  %gather1 = mo.gather(%input, %idx1) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32, dev:0>, !mo.tensor<[1], si64, dev:0>) -> !mo.tensor<[1, D0, 10], f32, dev:0>
  %gather2 = mo.gather(%input, %idx2) [axis = 0]
  : (!mo.tensor<[3, D0, 10], f32, dev:0>, !mo.tensor<[1], si64, dev:0>) -> !mo.tensor<[1, D0, 10], f32, dev:0>

  // CHECK-NEXT: %[[ADD01:.*]] = mo.add(%[[SPLIT_RES]]#0, %[[SPLIT_RES]]#1) : !mo.tensor<[1, [[D0]], 10], f32, dev:0>
  %add01 = mo.add(%gather0, %gather1) : !mo.tensor<[1, D0, 10], f32, dev:0>
  // CHECK-NEXT: %[[ADD012:.*]] = mo.add(%[[ADD01]], %[[SPLIT_RES]]#2) : !mo.tensor<[1, [[D0]], 10], f32, dev:0>
  %add012 = mo.add(%add01, %gather2) : !mo.tensor<[1, D0, 10], f32, dev:0>

  // CHECK-NEXT: mo.output %[[ADD012]] : !mo.tensor<[1, [[D0]], 10], f32, dev:0>
  mo.output %add012 : !mo.tensor<[1, D0, 10], f32, dev:0>
}

// CHECK: mo.graph @gather_to_split_dyn_axis_0<[[D0:.*]]>(
mo.graph @gather_to_split_dyn_axis_0<D0>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 3], f32>
  %input: !mo.tensor<[D0, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 3], f32> {
) -> !mo.tensor<[D0, 3], f32> {
  // CHECK-NEXT: %[[IDX0:.*]] = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  // CHECK-NEXT: %[[IDX1:.*]] = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx1 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  // CHECK-NEXT: %[[GATHER0:.*]] = mo.gather(%[[INPUT]], %[[IDX0]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[D0]], 3], f32>
  %gather0 = mo.gather(%input, %idx0) [axis = 0]
  : (!mo.tensor<[D0, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D0, 3], f32>
  // CHECK-NEXT: %[[GATHER1:.*]] = mo.gather(%[[INPUT]], %[[IDX1]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[D0]], 3], f32>
  %gather1 = mo.gather(%input, %idx1) [axis = 0]
  : (!mo.tensor<[D0, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D0, 3], f32>

  // CHECK-NEXT: %[[ADD:.*]] = mo.add(%[[GATHER0]], %[[GATHER1]]) : !mo.tensor<[[[D0]], 3], f32>
  %add = mo.add(%gather0, %gather1) : !mo.tensor<[D0, 3], f32>

  // CHECK-NEXT: mo.output %[[ADD]] : !mo.tensor<[[[D0]], 3], f32>
  mo.output %add : !mo.tensor<[D0, 3], f32>
}

// CHECK: mo.graph @gather_to_split_control_flow(
mo.graph @gather_to_split_control_flow(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 3], f32>,
  %input: !mo.tensor<[2, 3], f32>,
  // CHECK-SAME: %[[COND:.*]]: !mo.tensor<[], bool>
  %cond: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[2, 3], f32> {
) -> !mo.tensor<[2, 3], f32> {
  // CHECK-NEXT: %[[IDX0:.*]] = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx0 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  // CHECK-NEXT: %[[IDX1:.*]] = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %idx1 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  // CHECK-NEXT: %[[GATHER0:.*]] = mo.gather(%[[INPUT]], %[[IDX0]]) [axis = 0]
  // CHECK-SAME: : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[2, 3], f32>
  %gather0 = mo.gather(%input, %idx0) [axis = 0]
  : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[2, 3], f32>

  // CHECK-NEXT: %[[RES:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], f32> {
  %res = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], f32> {
    // CHECK-NEXT: %[[GATHER1:.*]] = mo.gather(%[[INPUT]], %[[IDX1]]) [axis = 0]
    // CHECK-SAME: : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[2, 3], f32>
    %gather1 = mo.gather(%input, %idx1) [axis = 0]
    : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[2, 3], f32>
    // CHECK-NEXT: %[[ADD:.*]] = mo.add(%[[GATHER0]], %[[GATHER1]]) : !mo.tensor<[2, 3], f32>
    %add = mo.add(%gather0, %gather1) : !mo.tensor<[2, 3], f32>
    // CHECK-NEXT: mo.yield %[[ADD]] : !mo.tensor<[2, 3], f32>
    mo.yield %add : !mo.tensor<[2, 3], f32>

  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mo.yield %[[INPUT]] : !mo.tensor<[2, 3], f32>
    mo.yield %input : !mo.tensor<[2, 3], f32>
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK:      mo.graph @gather_empty(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[0], si64>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[0], si64>
// CHECK-SAME: ) -> !mo.tensor<[0], si64> {
mo.graph @gather_empty(
  %input: !mo.tensor<[0], si64>,
  %indices: !mo.tensor<[0], si64>
) -> !mo.tensor<[0], si64> {

  // CHECK-NEXT: %[[GATHER:.*]] = mo.gather(%[[INPUT]], %[[INDICES]]) [axis = 0] : (!mo.tensor<[0], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[0], si64>
  %gather = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[0], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[0], si64>

  // CHECK-NEXT: mo.output %[[GATHER]] : !mo.tensor<[0], si64>
  mo.output %gather : !mo.tensor<[0], si64>
}
