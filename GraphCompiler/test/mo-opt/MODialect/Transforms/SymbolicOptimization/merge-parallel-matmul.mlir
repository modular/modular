// RUN: mo-opt -split-input-file --mo-symbolic-optimize %s | FileCheck %s

// Weights that all swap their last two dims merge through that transpose: the
// weights concatenate on the rows they are stored in, one transpose feeds the
// merged matmul, and each matmul keeps reading its own columns.
//
// CHECK-LABEL: mo.graph @transposed_weights
// CHECK-DAG: %[[PERM:.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
// CHECK-DAG: %[[WEIGHTS:.*]] = mo.concat[0] (%{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[9, 2], f32>
// CHECK: %[[TRANSPOSED:.*]] = mo.transpose(%[[WEIGHTS]], %[[PERM]]) : (!mo.tensor<[9, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 9], f32>
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %[[TRANSPOSED]]) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 9], f32>) -> !mo.tensor<[4, 9], f32>
// CHECK: %[[FIRST:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 9], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32>
// CHECK: %[[SECOND:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 9], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32>
// CHECK: %[[THIRD:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 9], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32>
// CHECK: mo.add(%[[FIRST]], %[[SECOND]])
// CHECK: mo.add({{.*}}, %[[THIRD]])
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.transpose

mo.graph @transposed_weights(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 3], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w3 = mo.constant <#M.dense_array<3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w3t = mo.transpose(%w3, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm3 = mo.matmul(%arg0, %w3t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %sum12 = mo.add(%mm1, %mm2) : !mo.tensor<[4, 3], f32>
  %sum = mo.add(%sum12, %mm3) : !mo.tensor<[4, 3], f32>
  mo.output %sum : !mo.tensor<[4, 3], f32>
}

// -----

// A concat that takes every result back reads the merged matmul itself, so the
// merge needs no split and no copy.
//
// CHECK-LABEL: mo.graph @concat_of_all_results
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
// CHECK: mo.output %[[MERGED]]
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.split
// CHECK-NOT: mo.slice

mo.graph @concat_of_all_results(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 6], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %both = mo.concat[1] (%mm1, %mm2) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 6], f32>)
  mo.output %both : !mo.tensor<[4, 6], f32>
}

// -----

// A concat that takes a run of results back reads one slice over that run,
// not a copy of it.
//
// CHECK-LABEL: mo.graph @runs_of_results
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 12], f32>) -> !mo.tensor<[4, 12], f32>
// CHECK: %[[FIRST:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 12], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 6], f32>
// CHECK: %[[SECOND:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 12], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 6], f32>
// CHECK: mo.add(%[[FIRST]], %[[SECOND]])
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.slice(%[[MERGED]], {{.*}}) -> !mo.tensor<[4, 3], f32>

mo.graph @runs_of_results(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 6], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w3 = mo.constant <#M.dense_array<3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w4 = mo.constant <#M.dense_array<4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w3t = mo.transpose(%w3, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm3 = mo.matmul(%arg0, %w3t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w4t = mo.transpose(%w4, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm4 = mo.matmul(%arg0, %w4t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %left = mo.concat[1] (%mm1, %mm2) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 6], f32>)
  %right = mo.concat[1] (%mm3, %mm4) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 6], f32>)
  %sum = mo.add(%left, %right) : !mo.tensor<[4, 6], f32>
  mo.output %sum : !mo.tensor<[4, 6], f32>
}

// -----

// A weight defined after the earliest matmul moves above the merged one, so
// the concat never reads a value defined after it.
//
// CHECK-LABEL: mo.graph @weight_after_matmul
// CHECK: %[[WEIGHTS:.*]] = mo.concat[0] (
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
// CHECK-NOT: mo.matmul

mo.graph @weight_after_matmul(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 3], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %sum = mo.add(%mm1, %mm2) : !mo.tensor<[4, 3], f32>
  mo.output %sum : !mo.tensor<[4, 3], f32>
}

// -----

// A mixed group still merges: `B` is `[in, out]` with or without a transpose,
// and one transpose among the inputs keeps this concat out of the move above.
//
// CHECK-LABEL: mo.graph @mixed_weights
// CHECK: %[[WEIGHTS:.*]] = mo.concat[1] ({{.*}}) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 6], f32>
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %[[WEIGHTS]]) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
// CHECK-NOT: mo.matmul

mo.graph @mixed_weights(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 3], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %sum = mo.add(%mm1, %mm2) : !mo.tensor<[4, 3], f32>
  mo.output %sum : !mo.tensor<[4, 3], f32>
}

// -----

// A three-way concat reads one slice over the three columns, not a split of
// three results and a copy.
//
// CHECK-LABEL: mo.graph @run_of_three
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 18], f32>) -> !mo.tensor<[4, 18], f32>
// CHECK: %[[LEFT:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 18], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 9], f32>
// CHECK: %[[RIGHT:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 18], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 9], f32>
// CHECK: mo.add(%[[LEFT]], %[[RIGHT]])
// CHECK-NOT: mo.concat[1]

mo.graph @run_of_three(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 9], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w3 = mo.constant <#M.dense_array<3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w4 = mo.constant <#M.dense_array<4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00> : !M.array<18xf32>> : !mo.tensor<[9, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w3t = mo.transpose(%w3, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm3 = mo.matmul(%arg0, %w3t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w4t = mo.transpose(%w4, %perm) : (!mo.tensor<[9, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 9], f32>
  %mm4 = mo.matmul(%arg0, %w4t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 9], f32>) -> !mo.tensor<[4, 9], f32>
  %left = mo.concat[1] (%mm1, %mm2, %mm3) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 9], f32>
  %sum = mo.add(%left, %mm4) : !mo.tensor<[4, 9], f32>
  mo.output %sum : !mo.tensor<[4, 9], f32>
}

// -----

// `nn.Linear` writes `x @ w.T + 0`, so the results reach the concat through a
// splat zero until the zero-add fold runs. The merge reads through it.
//
// CHECK-LABEL: mo.graph @zero_added_results
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
// CHECK: mo.output %[[MERGED]]
// CHECK-NOT: mo.add
// CHECK-NOT: mo.concat[1]

mo.graph @zero_added_results(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 6], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %zero = mo.constant <#M.dense_array<0.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
  %bcast_zero = mo.static.broadcast_to(%zero) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %a1 = mo.add(%mm1, %bcast_zero) : !mo.tensor<[4, 3], f32>
  %a2 = mo.add(%mm2, %bcast_zero) : !mo.tensor<[4, 3], f32>
  %both = mo.concat[1] (%a1, %a2) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 6], f32>
  mo.output %both : !mo.tensor<[4, 6], f32>
}

// -----

// A batch matmul concatenates its weights one dim earlier too: on axis 1 of
// [batch, out, in], then the transpose of the last two dims.
//
// CHECK-LABEL: mo.graph @batch_matmul_transposed_weights
// CHECK: %[[WEIGHTS:.*]] = mo.concat[1] (
// CHECK: %[[MERGED:.*]] = mo.batch_matmul(%arg0, %{{.*}}) : (!mo.tensor<[4, 5, 2], f32>, !mo.tensor<[4, 2, 6], f32>) -> !mo.tensor<[4, 5, 6], f32>
// CHECK: %[[FIRST:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 5, 6], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 5, 3], f32>
// CHECK: %[[SECOND:.*]] = mo.slice(%[[MERGED]], %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[4, 5, 6], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 5, 3], f32>
// CHECK: mo.add(%[[FIRST]], %[[SECOND]])

mo.graph @batch_matmul_transposed_weights(%arg0: !mo.tensor<[4, 5, 2], f32>) -> !mo.tensor<[4, 5, 3], f32> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<12xf32>> : !mo.tensor<[4, 3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<12xf32>> : !mo.tensor<[4, 3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[4, 2, 3], f32>
  %mm1 = mo.batch_matmul(%arg0, %w1t) : (!mo.tensor<[4, 5, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 5, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[4, 2, 3], f32>
  %mm2 = mo.batch_matmul(%arg0, %w2t) : (!mo.tensor<[4, 5, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 5, 3], f32>
  %sum = mo.add(%mm1, %mm2) : !mo.tensor<[4, 5, 3], f32>
  mo.output %sum : !mo.tensor<[4, 5, 3], f32>
}

// -----

// Weights already packed with `transpose_b` concatenate on their column axis,
// so they are left alone rather than merged on the wrong one.
//
// CHECK-LABEL: mo.graph @transposed_b
// CHECK: mo.matmul({{.*}}) [transpose_b = true]
// CHECK: mo.matmul({{.*}}) [transpose_b = true]
// CHECK-NOT: mo.split
// CHECK-NOT: mo.concat[0]

mo.graph @transposed_b(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 3], f32> {
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32, {layout = #mo.layout<NK>}>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32, {layout = #mo.layout<NK>}>
  %mm1 = mo.matmul(%arg0, %w1) [transpose_b = true] : (!mo.tensor<[4, 2], f32>, !mo.tensor<[3, 2], f32, {layout = #mo.layout<NK>}>) -> !mo.tensor<[4, 3], f32>
  %mm2 = mo.matmul(%arg0, %w2) [transpose_b = true] : (!mo.tensor<[4, 2], f32>, !mo.tensor<[3, 2], f32, {layout = #mo.layout<NK>}>) -> !mo.tensor<[4, 3], f32>
  %sum = mo.add(%mm1, %mm2) : !mo.tensor<[4, 3], f32>
  mo.output %sum : !mo.tensor<[4, 3], f32>
}

// -----

// One matmul has nothing to merge, but its concat of transposes moves before
// them: one transpose over the concatenated operands.
//
// CHECK-LABEL: mo.graph @concat_of_transposes
// CHECK: %[[WEIGHTS:.*]] = mo.concat[0] ({{.*}}, {{.*}}) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[6, 2], f32>
// CHECK: %[[TRANSPOSED:.*]] = mo.transpose(%[[WEIGHTS]], %{{.*}}) : (!mo.tensor<[6, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 6], f32>
// CHECK: %[[MERGED:.*]] = mo.matmul(%arg0, %[[TRANSPOSED]]) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
// CHECK-NOT: mo.transpose
// CHECK-NOT: mo.concat[1]

mo.graph @concat_of_transposes(%arg0: !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 6], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], f32>
  %both = mo.concat[1] (%w1t, %w2t) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 6], f32>)
  %mm = mo.matmul(%arg0, %both) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 6], f32>) -> !mo.tensor<[4, 6], f32>
  mo.output %mm : !mo.tensor<[4, 6], f32>
}

// -----

// The axis follows the permutation, it is not just the last-two-dims swap:
// with `perm = [1, 2, 0]`, output dim 1 is input dim 2, so a concat on output
// dim 1 moves to input dim 2.
//
// CHECK-LABEL: mo.graph @cyclic_permutation
// CHECK: %[[WEIGHTS:.*]] = mo.concat[2] ({{.*}}, {{.*}}) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 8], f32>
// CHECK: mo.transpose(%[[WEIGHTS]], {{.*}}) : (!mo.tensor<[2, 3, 8], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 8, 2], f32>
// CHECK-NOT: mo.concat[1]

mo.graph @cyclic_permutation(%a: !mo.tensor<[2, 3, 4], f32>, %b: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[3, 8, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 2, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %at = mo.transpose(%a, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %bt = mo.transpose(%b, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %both = mo.concat[1] (%at, %bt) : (!mo.tensor<[3, 4, 2], f32>, !mo.tensor<[3, 4, 2], f32>) -> (!mo.tensor<[3, 8, 2], f32>)
  mo.output %both : !mo.tensor<[3, 8, 2], f32>
}

// -----

// Matmuls that differ in result dtype stay apart: the merged matmul would have
// one result dtype.
//
// CHECK-LABEL: mo.graph @mixed_result_dtypes
// CHECK: mo.matmul({{.*}}) : (!mo.tensor<[4, 2], bf16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[4, 3], f32>
// CHECK: mo.matmul({{.*}}) : (!mo.tensor<[4, 2], bf16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[4, 3], bf16>
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.split

mo.graph @mixed_result_dtypes(%arg0: !mo.tensor<[4, 2], bf16>) -> (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], bf16>) {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w1 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : !M.array<6xbf16>> : !mo.tensor<[3, 2], bf16>
  %w2 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : !M.array<6xbf16>> : !mo.tensor<[3, 2], bf16>
  %w1t = mo.transpose(%w1, %perm) : (!mo.tensor<[3, 2], bf16>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], bf16>
  %mm1 = mo.matmul(%arg0, %w1t) : (!mo.tensor<[4, 2], bf16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[4, 3], f32>
  %w2t = mo.transpose(%w2, %perm) : (!mo.tensor<[3, 2], bf16>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 3], bf16>
  %mm2 = mo.matmul(%arg0, %w2t) : (!mo.tensor<[4, 2], bf16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[4, 3], bf16>
  %r1 = mo.mul(%mm1, %mm1) : !mo.tensor<[4, 3], f32>
  %r2 = mo.mul(%mm2, %mm2) : !mo.tensor<[4, 3], bf16>
  mo.output %r1, %r2 : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], bf16>
}
