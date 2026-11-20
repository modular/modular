// RUN: mo-opt %s --mo-splat-large-constants=num-elements=4 --cse --canonicalize --mo-shape-inference --mo-symbolic-optimize --cse --canonicalize | FileCheck %s

// CHECK-LABEL:  mo.graph @base_case
// CHECK-NEXT:    %[[CST:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 3], si32>
// CHECK-NEXT:    mo.output %[[BCAST]] : !mo.tensor<[2, 3], si32>
mo.graph @base_case() -> (!mo.tensor<[2, 3], si32>) {
  %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  mo.output %cst : !mo.tensor<[2, 3], si32>
}

// CHECK-LABEL:  mo.graph @control_flow
// CHECK-SAME:    (%[[COND:.*]]: !mo.tensor<[], bool>)
// CHECK-NEXT:    %[[CST:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:    %[[RET:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> {
// CHECK-NEXT:      %[[BCAST0:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 3], si32>
// CHECK-NEXT:      mo.yield %[[BCAST0]] : !mo.tensor<[2, 3], si32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[BCAST1:.*]] = mo.static.broadcast_to(%[[CST]]) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 3], si32>
// CHECK-NEXT:      mo.yield %[[BCAST1]] : !mo.tensor<[2, 3], si32>
// CHECK-NEXT:    }
// CHECK-NEXT:    mo.output %[[RET]] : !mo.tensor<[2, 3], si32>
mo.graph @control_flow(%cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2, 3], si32>) {
  %ret = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  } else {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  }
  mo.output %ret : !mo.tensor<[2, 3], si32>
}

// CHECK-LABEL:  mo.graph @multiple_constants
// CHECK-NEXT:    %[[CST2:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi16>> : !mo.tensor<[1], si16>
// CHECK-NEXT:    %[[CST1:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-NEXT:    %[[CST0:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:    %[[BCAST0:.*]] = mo.static.broadcast_to(%[[CST0]]) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 3], si32>
// CHECK-NEXT:    %[[BCAST1:.*]] = mo.static.broadcast_to(%[[CST1]]) : !mo.tensor<[1], si64> -> !mo.tensor<[2, 3], si64>
// CHECK-NEXT:    %[[BCAST2:.*]] = mo.static.broadcast_to(%[[CST2]]) : !mo.tensor<[1], si16> -> !mo.tensor<[2, 3], si16>
// CHECK-NEXT:    mo.output %[[BCAST0]], %[[BCAST1]], %[[BCAST2]] : !mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si16>
mo.graph @multiple_constants() -> (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si16>) {
  // Different data types
  %cst0 = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %cst1 = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi64>> : !mo.tensor<[2, 3], si64>
  %cst2 = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi16>> : !mo.tensor<[2, 3], si16>
  mo.output %cst0, %cst1, %cst2 : !mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si16>
}

// CHECK-LABEL: mo.graph @too_small
// CHECK-NEXT:   %[[CST:.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
// CHECK-NEXT:   mo.output %[[CST]] : !mo.tensor<[2], si32>
mo.graph @too_small() -> (!mo.tensor<[2], si32>) {
  %cst = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  mo.output %cst : !mo.tensor<[2], si32>
}

// This tests that splatted constants survive in their splatted form through
// our shape optimization pipeline.

// CHECK-LABEL: mo.graph @preserve_splat
mo.graph @preserve_splat() -> !mo.tensor<[288], f32> {
    %0 = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
    %1 = mo.constant <#M.dense_array<288> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %2 = mo.broadcast_to(%0, %1) : (!mo.tensor<[], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[288], f32>

    // CHECK-NEXT: %[[SPLAT:.*]] = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
    // CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[SPLAT]]) : !mo.tensor<[], f32> -> !mo.tensor<[288], f32>
    // CHECK-NEXT: mo.output %[[BCAST]]
    mo.output %2 : !mo.tensor<[288], f32>
}
