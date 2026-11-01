// RUN: mo-opt %s --canonicalize | FileCheck %s

// COM: A mean is a sum divided by the length of the reduced axis.
// CHECK: mo.graph @mean(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32> {
mo.graph @mean(%input: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 1], f32>) {
  // CHECK: %[[SUM:.*]] = mo.reduce.add(%[[INPUT]]) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
  // CHECK: mo.div(%[[SUM]], %{{.*}}) : !mo.tensor<[2, 1], f32>
  %sum = mo.reduce.add(%input) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
  %len = mo.constant <#M.dense_array<3.0> : !M.array<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[2, 1], f32>
  %output = mo.div(%sum, %len_b) : !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// CHECK: mo.graph @mean_middle_axis(%[[INPUT:.*]]: !mo.tensor<[2, 3, 5], f32>) -> !mo.tensor<[2, 1, 5], f32> {
mo.graph @mean_middle_axis(%input: !mo.tensor<[2, 3, 5], f32>) -> (!mo.tensor<[2, 1, 5], f32>) {
  // CHECK: %[[SUM:.*]] = mo.reduce.add(%[[INPUT]]) : (!mo.tensor<[2, 3, 5], f32>) -> !mo.tensor<[2, 1, 5], f32>
  // CHECK: mo.div(%[[SUM]], %{{.*}}) : !mo.tensor<[2, 1, 5], f32>
  %sum = mo.reduce.add(%input) : (!mo.tensor<[2, 3, 5], f32>) -> !mo.tensor<[2, 1, 5], f32>
  %len = mo.constant <#M.dense_array<3.0> : !M.array<1xf32>> : !mo.tensor<[], f32>
  %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[2, 1, 5], f32>
  %output = mo.div(%sum, %len_b) : !mo.tensor<[2, 1, 5], f32>
  mo.output %output : !mo.tensor<[2, 1, 5], f32>
}

// CHECK: mo.graph @reduce_mul(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_mul(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.mul(%[[INPUT]]) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.mul(%input) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_mul_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_mul_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.mul(%[[INPUT]]) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.mul(%input) [axis = -1] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_add(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_add(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.add(%[[INPUT]]) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_add_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[5, ?, ?], f32>) -> !mo.tensor<[1, ?, ?], f32> {
mo.graph @reduce_add_negative_index_canonicalization(%input: !mo.tensor<[5, ?, ?], f32>) -> (!mo.tensor<[1, ?, ?], f32>) {
  // CHECK-NEXT: mo.reduce.add(%[[INPUT]]) : (!mo.tensor<[5, ?, ?], f32>) -> !mo.tensor<[1, ?, ?], f32>
  %output = mo.reduce.add(%input) : (!mo.tensor<[5, ?, ?], f32>) -> !mo.tensor<[1, ?, ?], f32>
  mo.output %output : !mo.tensor<[1, ?, ?], f32>
}

// CHECK: mo.graph @reduce_add_fold(%[[INPUT:.*]]: !mo.tensor<[?, ?, 1], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_add_fold(%input: !mo.tensor<[?, ?, 1], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: mo.output %[[INPUT:.*]] : !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 1], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_add_no_fold(%[[INPUT:.*]]: !mo.tensor<[?, ?, 10], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_add_no_fold(%input: !mo.tensor<[?, ?, 10], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // Make sure the reduction is not removed since the input reduction dimension is not 1
  // CHECK: mo.reduce.add(%[[INPUT]]) : (!mo.tensor<[?, ?, 10], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.add(%input) : (!mo.tensor<[?, ?, 10], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_min(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_min(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.min(%[[INPUT]]) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.min(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_min_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_min_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.min(%[[INPUT]]) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.min(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_max(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_max(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.max(%[[INPUT]]) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.max(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_max_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @reduce_max_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK-NEXT: mo.reduce.max(%[[INPUT]]) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.max(%input) : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @arg_max(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64> {
mo.graph @arg_max(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], si64>) {
  // CHECK-NEXT: mo.reduce.arg_max(%[[INPUT]]) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64>
  %output = mo.reduce.arg_max(%input) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64>
  mo.output %output : !mo.tensor<[?, ?, 1], si64>
}

// CHECK: mo.graph @arg_min(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64> {
mo.graph @arg_min(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], si64>) {
  // CHECK-NEXT: mo.reduce.arg_min(%[[INPUT]]) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64>
  %output = mo.reduce.arg_min(%input) [axis = 2] : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64>
  mo.output %output : !mo.tensor<[?, ?, 1], si64>
}
