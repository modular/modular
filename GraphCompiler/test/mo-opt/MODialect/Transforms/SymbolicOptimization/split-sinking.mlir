// RUN: mo-opt --mo-symbolic-optimize %s -split-input-file | FileCheck %s

/////////////////////////////////
////////// ELEMENTWISE //////////
/////////////////////////////////

// CHECK-LABEL: mo.graph @unary_elementwise
// CHECK-NEXT:   %[[SIZES:.*]] = mo.constant
// CHECK-NEXT:   %[[EXP:.*]] = mo.exp({{.*}}) : !mo.tensor<[2, 4], f32>
// CHECK-NEXT:   %[[SPLIT:.*]]:2 = mo.split(%[[EXP]], %[[SIZES]]) [axis = 1]
// CHECK-NEXT:   mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>

mo.graph @unary_elementwise(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.exp(%2#0) : !mo.tensor<[2, 2], f32>
  %4 = mo.exp(%2#1) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @unary_elementwise_with_device
// CHECK-NEXT:   %[[SIZES:.*]] = mo.constant
// CHECK-NEXT:   %[[EXP:.*]] = mo.exp({{.*}}) : !mo.tensor<[2, 4], f32, dev:0>
// CHECK-NEXT:   %[[SPLIT:.*]]:2 = mo.split(%[[EXP]], %[[SIZES]]) [axis = 1]
// CHECK-NEXT:   mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>

mo.graph @unary_elementwise_with_device(%in: !mo.tensor<[2, 4], f32, dev:0>) -> (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32, dev:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>)
  %3 = mo.exp(%2#0) : !mo.tensor<[2, 2], f32, dev:0>
  %4 = mo.exp(%2#1) : !mo.tensor<[2, 2], f32, dev:0>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>
}

// CHECK-LABEL:  mo.graph @binary_elementwise
// CHECK-DAG:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-DAG:    %[[CST:.*]] = mo.constant <#M.dense_array<0.000000e+00
// CHECK:    %[[CONCAT:.*]] = mo.concat[1] (%[[CST]], %[[CST]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 4], f32>
// CHECK-NEXT:    %[[ADD:.*]] = mo.add(%{{.*}}, %[[CONCAT]]) : !mo.tensor<[2, 4], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[ADD]], %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>

mo.graph @binary_elementwise(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst = mo.constant <#M.dense_array<0., 1., 2., 3.> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.add(%2#0, %cst) : !mo.tensor<[2, 2], f32>
  %4 = mo.add(%2#1, %cst) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @binary_elementwise_with_broadcast
// CHECK-DAG:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-DAG:    %[[CST:.*]] = mo.constant <#M.dense_array<0.000000e+00, 1.000000e+00> : tensor<2xf32>> : !mo.tensor<[2], f32>
// CHECK:    %[[CONCAT:.*]] = mo.concat[0] (%[[CST]], %[[CST]])
// CHECK-NEXT:    %[[BCAST:.*]] = mo.static.broadcast_to(%[[CONCAT]])
// CHECK-NEXT:    %[[ADD:.*]] = mo.add(%{{.*}}, %[[BCAST]]) : !mo.tensor<[2, 4], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[ADD]], %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>

mo.graph @binary_elementwise_with_broadcast(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst = mo.constant <#M.dense_array<0., 1.> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[2], f32> -> !mo.tensor<[2, 2], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.add(%2#0, %bcast) : !mo.tensor<[2, 2], f32>
  %4 = mo.add(%2#1, %bcast) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_sinking_binary_elementwise_different_index
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.sub
// CHECK-NEXT:    mo.sub

mo.graph @no_sinking_binary_elementwise_different_index(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst = mo.constant <#M.dense_array<0., 1., 2., 3.> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.sub(%2#0, %cst) : !mo.tensor<[2, 2], f32>
  %4 = mo.sub(%cst, %2#1) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_sinking_binary_elementwise_different_ops
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.sub
// CHECK-NEXT:    mo.add

mo.graph @no_sinking_binary_elementwise_different_ops(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst = mo.constant <#M.dense_array<0., 1., 2., 3.> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.sub(%2#0, %cst) : !mo.tensor<[2, 2], f32>
  %4 = mo.add(%cst, %2#1) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_sinking_binary_elementwise_diff_broadcast
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.add
// CHECK-NEXT:    mo.add

mo.graph @no_sinking_binary_elementwise_diff_broadcast(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst1 = mo.constant <#M.dense_array<0., 1.> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %cst2 = mo.constant <#M.dense_array<2.> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %bcast1 = mo.static.broadcast_to(%cst1) : !mo.tensor<[2], f32> -> !mo.tensor<[2, 2], f32>
  %bcast2 = mo.static.broadcast_to(%cst2) : !mo.tensor<[1], f32> -> !mo.tensor<[2, 2], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.add(%2#0, %bcast1) : !mo.tensor<[2, 2], f32>
  %4 = mo.add(%2#1, %bcast2) : !mo.tensor<[2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_sinking_binary_elementwise_scalar
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.add
// CHECK-NEXT:    mo.add

mo.graph @no_sinking_binary_elementwise_scalar(%in: !mo.tensor<[2, 2000], f32>) -> (!mo.tensor<[2, 1000], f32>, !mo.tensor<[2, 1000], f32>) {
  %1 = mo.constant <#M.dense_array<1000, 1000> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %cst = mo.constant <#M.dense_array<1.> : tensor<f32>> : !mo.tensor<[], f32>
  %bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[2, 1000], f32>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 2000], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 1000], f32>, !mo.tensor<[2, 1000], f32>)
  %3 = mo.add(%2#0, %bcast) : !mo.tensor<[2, 1000], f32>
  %4 = mo.add(%2#1, %bcast) : !mo.tensor<[2, 1000], f32>
  mo.output %3, %4 : !mo.tensor<[2, 1000], f32>, !mo.tensor<[2, 1000], f32>
}

// -----

/////////////////////////////////
//////// RESHAPE/REBIND /////////
/////////////////////////////////

// CHECK-LABEL:  mo.graph @reshape_last_dim
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 4, 2], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>

mo.graph @reshape_last_dim(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[2, 4], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_not_last_axis
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8, 2], f32> -> !mo.tensor<[2, 4, 2, 2], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4, 2, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2, 2, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2, 2, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>

mo.graph @reshape_not_last_axis(%in: !mo.tensor<[2, 8, 2], f32>) -> (!mo.tensor<[2, 2, 2, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 8, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[2, 4, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 4, 2], f32> -> !mo.tensor<[2, 2, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 4, 2], f32> -> !mo.tensor<[2, 2, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2, 2, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_1_prefix
// CHECK-DAG:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[1, 2, 4], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 2] : (!mo.tensor<[1, 2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 2], f32>

mo.graph @reshape_1_prefix(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_1_in_middle
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8, 2], f32> -> !mo.tensor<[2, 4, 1, 1, 2, 2], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4, 1, 1, 2, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2, 1, 1, 2, 2], f32>, !mo.tensor<[2, 2, 1, 1, 2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2, 1, 1, 2, 2], f32>, !mo.tensor<[2, 2, 1, 1, 2, 2], f32>

mo.graph @reshape_1_in_middle(%in: !mo.tensor<[2, 8, 2], f32>) -> (!mo.tensor<[2, 2, 1, 1, 2, 2], f32>, !mo.tensor<[2, 2, 1, 1, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 8, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[2, 4, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 4, 2], f32> -> !mo.tensor<[2, 2, 1, 1, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 4, 2], f32> -> !mo.tensor<[2, 2, 1, 1, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2, 1, 1, 2, 2], f32>, !mo.tensor<[2, 2, 1, 1, 2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_mixed_but_trailing_dimensions
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<4, 4> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[4, 2], f32> -> !mo.tensor<[8], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 0] : (!mo.tensor<[8], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>

mo.graph @reshape_mixed_but_trailing_dimensions(%in: !mo.tensor<[4, 2], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 0] : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  mo.output %3, %4 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_mixed_but_look_right
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<100, 100> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 100], f32> -> !mo.tensor<[200], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[RESHAPE]], %[[SIZES]]) [axis = 0] : (!mo.tensor<[200], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[100], f32>, !mo.tensor<[100], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[100], f32>, !mo.tensor<[100], f32>

mo.graph @reshape_mixed_but_look_right(%in: !mo.tensor<[2, 100], f32>) -> (!mo.tensor<[100], f32>, !mo.tensor<[100], f32>) {
  %1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 0] : (!mo.tensor<[2, 100], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[1, 100], f32>, !mo.tensor<[1, 100], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[1, 100], f32> -> !mo.tensor<[100], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[1, 100], f32> -> !mo.tensor<[100], f32>
  mo.output %3, %4 : !mo.tensor<[100], f32>, !mo.tensor<[100], f32>
}

// CHECK-LABEL:  mo.graph @reshape_no_sinking_1_suffix_mixed_dimensions
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.static.reshape
// CHECK-NEXT:    mo.static.reshape

mo.graph @reshape_no_sinking_1_suffix_mixed_dimensions(%in: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 1], f32>, !mo.tensor<[2, 1], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2], f32>
  mo.output %3, %4 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

// The two slices have different reshape targets: [1, 2, 2] can swim (1-to-1
// mapping for the sliced axis), while [4] cannot (mixing). The first case
// is transformed by SwimStaticReshapeLikeOpsThroughSliceOp.
// CHECK-LABEL:  mo.graph @no_sinking_different_reshapes
// CHECK:         mo.static.reshape
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.static.reshape

mo.graph @no_sinking_different_reshapes(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[4], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  mo.output %3, %4 : !mo.tensor<[1, 2, 2], f32>, !mo.tensor<[4], f32>
}

// CHECK-LABEL:  mo.graph @no_sinking_dimensions_are_mixed
// CHECK:         mo.slice
// CHECK-NEXT:    mo.slice
// CHECK-NEXT:    mo.static.reshape
// CHECK-NEXT:    mo.static.reshape

mo.graph @no_sinking_dimensions_are_mixed(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  mo.output %3, %4 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// COM: 0 element tensors have ambiguous mappings between before and after dims
// CHECK-LABEL: mo.graph
// CHECK-SAME: @no_sinking_reshape_0ele([[VARG0:[^ ]*]]: !mo.tensor<[0, 8], f32>) -> (!mo.tensor<[0, 2, 2], f32>, !mo.tensor<[0, 2, 2], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<9223372036854775807, 8> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<0, 4> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.constant <#M.dense_array<0, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.constant <#M.dense_array<9223372036854775807, 4> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.slice([[VARG0]], [[V3]], [[V4]], [[V2]]) : (!mo.tensor<[0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[0, 4], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.slice([[VARG0]], [[V1]], [[V0]], [[V2]]) : (!mo.tensor<[0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[0, 4], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.static.reshape([[V5]]) : !mo.tensor<[0, 4], f32> -> !mo.tensor<[0, 2, 2], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.static.reshape([[V6]]) : !mo.tensor<[0, 4], f32> -> !mo.tensor<[0, 2, 2], f32>
// CHECK-NEXT: mo.output [[V7]], [[V8]] : !mo.tensor<[0, 2, 2], f32>, !mo.tensor<[0, 2, 2], f32>
mo.graph @no_sinking_reshape_0ele(%in: !mo.tensor<[0, 8], f32>) -> (!mo.tensor<[0, 2, 2], f32>, !mo.tensor<[0, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 1] : (!mo.tensor<[0, 8], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[0, 4], f32>, !mo.tensor<[0, 4], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[0, 4], f32> -> !mo.tensor<[0, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[0, 4], f32> -> !mo.tensor<[0, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[0, 2, 2], f32>, !mo.tensor<[0, 2, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @rebind
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[REBIND:.*]] = mo.rebind({{.*}}) : !mo.tensor<[D0, 2, 4], f32> -> !mo.tensor<[D1, 2, 4], f32>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%[[REBIND]], %[[SIZES]]) [axis = 2] : (!mo.tensor<[D1, 2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[D1, 2, 2], f32>, !mo.tensor<[D1, 2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[D1, 2, 2], f32>, !mo.tensor<[D1, 2, 2], f32>

mo.graph @rebind<D0, D1>(%in: !mo.tensor<[D0, 2, 4], f32>) -> (!mo.tensor<[D1, 2, 2], f32>, !mo.tensor<[D1, 2, 2], f32>) {
  %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2:2 = mo.split(%in, %1) [axis = 2] : (!mo.tensor<[D0, 2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[D0, 2, 2], f32>, !mo.tensor<[D0, 2, 2], f32>)
  %3 = mo.rebind(%2#0) : !mo.tensor<[D0, 2, 2], f32> -> !mo.tensor<[D1, 2, 2], f32>
  %4 = mo.rebind(%2#1) : !mo.tensor<[D0, 2, 2], f32> -> !mo.tensor<[D1, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[D1, 2, 2], f32>, !mo.tensor<[D1, 2, 2], f32>
}
