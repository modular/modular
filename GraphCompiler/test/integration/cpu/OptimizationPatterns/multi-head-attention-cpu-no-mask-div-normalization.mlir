// Check pattern works with div normalization
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// Check E2E performance. Note we also run the case where the pattern does not
// trigger (due to improper devices in graph) E2E to help establish correctness.

// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute %t.mlir --input-shapes=1x8x2x4 --input-shapes=1x8x2x4 --input-shapes=1x8x2x4 --input-shapes=1 --model-inputs=random --result-output-style=full | FileCheck %s -check-prefix=CHECK-MT

// Dimension names:
//     - (B)atch
//     - Attention (H)ead
//     - (S)equence
//     - Embedding (D)imension
mo.graph @cpu_flash_attn_pattern<B, H, S, D>(
    %query_pre_transpose: !mo.tensor<[B, S, H, D], f32>,
    %key_pre_transpose: !mo.tensor<[B, S, H, D], f32>,
    %value_pre_transposed: !mo.tensor<[B, S, H, D], f32>,
    %norm_scalar1: !mo.tensor<[1], f32>
) -> !mo.tensor<[B, H, S, D], f32> {
    %perm1 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %norm_scalar = mo.static.reshape(%norm_scalar1): !mo.tensor<[1], f32> -> !mo.tensor<[], f32>

    // use for normalizing attnMatrix (attn_mat)
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[B, H, S, S], f32>

    // prepare qkv in the inner loop
    %query = mo.transpose(%query_pre_transpose, %perm1) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
    %keyT = mo.transpose(%key_pre_transpose, %perm2) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, D, S], f32>
    %value = mo.transpose(%value_pre_transposed, %perm1) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>

    // the actual calculation, attn is pretty simple huh?
    %attn_mat = mo.batch_matmul(%query, %keyT) : (!mo.tensor<[B, H, S, D], f32>, !mo.tensor<[B, H, D, S], f32>) -> !mo.tensor<[B, H, S, S], f32>
    %attn_mat_norm = mo.div(%attn_mat, %norm) : !mo.tensor<[B, H, S, S], f32>
    %attn_mat_softmax = mo.reduce.softmax(%attn_mat_norm) [axis = 3] : (!mo.tensor<[B, H, S, S], f32>) -> !mo.tensor<[B, H, S, S], f32>
    %reproj = mo.batch_matmul(%attn_mat_softmax, %value) : (!mo.tensor<[B, H, S, S], f32>, !mo.tensor<[B, H, S, D], f32>) -> !mo.tensor<[B, H, S, D], f32>

    mo.output %reproj: !mo.tensor<[B, H, S, D], f32>
}
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK-MT: --- Running 'cpu_flash_attn_pattern':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT:[0-9\-\.e ,]+]]
// CHECK-MT: ]

// CHECK: mo.graph @cpu_flash_attn_pattern<B, H, S, D>(
// CHECK-SAME: [[QUERY:%.[a-zA-Z0-9_]*]]: !mo.tensor<[B, S, H, D], f32>,
// CHECK-SAME: [[KEYT:%.[a-zA-Z0-9_]*]]: !mo.tensor<[B, S, H, D], f32>,
// CHECK-SAME: [[VALUE:%.[a-zA-Z0-9_]*]]: !mo.tensor<[B, S, H, D], f32>,
// CHECK-SAME: [[NORM:%.[a-zA-Z0-9_]*]]: !mo.tensor<[1], f32>)
// CHECK-SAME: -> !mo.tensor<[B, H, S, D], f32> {
// CHECK-NEXT: [[V0:%.[a-zA-Z0-9_]*]] = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf64>> : !mo.tensor<[], f64>
// CHECK-NEXT: [[V1:%.*]] = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V2:%.[a-zA-Z0-9_]*]] = mo.static.reshape([[NORM]]) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:%.[a-zA-Z0-9_]*]] = mo.cast([[V0]]) : (!mo.tensor<[], f64>) -> !mo.tensor<[], f32>
// CHECK-NEXT: [[V4:%.[a-zA-Z0-9_]*]] = mo.static.broadcast_to([[V3]]) : !mo.tensor<[], f32> -> !mo.tensor<[], f32>
// CHECK-NEXT: [[V5:%.[a-zA-Z0-9_]*]] = mo.div([[V4]], [[V2]]) : !mo.tensor<[], f32>
// CHECK-NEXT: [[V6:%.[a-zA-Z0-9_]*]] = mo.composite.no_mask_flash_attention_cpu([[QUERY]], [[KEYT]], [[VALUE]], [[V5]]) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[B, S, H, D], f32>, !mo.tensor<[B, S, H, D], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[B, S, H, D], f32>
// CHECK-NEXT: [[V7:%.*]] = mo.transpose([[V6]], [[V1]]) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
// CHECK-NEXT: mo.output [[V7]] : !mo.tensor<[B, H, S, D], f32>

// has the wrong device ("cpu2", want "cpu") so pattern won't fire
mo.graph @cpu_flash_attn_not_pattern<B, H, S, D> (
    %query_pre_transpose: !mo.tensor<[B, S, H, D], f32>,
    %key_pre_transpose: !mo.tensor<[B, S, H, D], f32>,
    %value_pre_transposed: !mo.tensor<[B, S, H, D], f32>,
    %norm_scalar1: !mo.tensor<[1], f32>
) -> !mo.tensor<[B, H, S, D], f32>
{
    %perm1 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    // NOTE: this manually does the implemented conversion of div --> mul to maintain the same precision as above
    %one = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
    %norm_scalar2 = mo.div(%one, %norm_scalar1) : !mo.tensor<[1], f32>
    %norm_scalar = mo.static.reshape(%norm_scalar2): !mo.tensor<[1], f32> -> !mo.tensor<[], f32>

    // use for normalizing attnMatrix (attn_mat)
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[mul_no_wrap(B, H), S, S], f32>

    // prepare qkv in the inner loop
    %query = mo.transpose(%query_pre_transpose, %perm1) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
    %query_reshaped = mo.static.reshape(%query) : !mo.tensor<[B, H, S, D], f32> -> !mo.tensor<[mul_no_wrap(B, H), S, D], f32>

    %keyT = mo.transpose(%key_pre_transpose, %perm2) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, D, S], f32>
    %keyT_reshaped = mo.static.reshape(%keyT) : !mo.tensor<[B, H, D, S], f32> -> !mo.tensor<[mul_no_wrap(B, H), D, S], f32>

    %value = mo.transpose(%value_pre_transposed, %perm1) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
    %value_reshaped = mo.static.reshape(%value) : !mo.tensor<[B, H, S, D], f32> -> !mo.tensor<[mul_no_wrap(B, H), S, D], f32>

    // the actual calculation, attn is pretty simple huh?
    %attn_mat = mo.batch_matmul(%query_reshaped, %keyT_reshaped) : (!mo.tensor<[mul_no_wrap(B, H), S, D], f32>, !mo.tensor<[mul_no_wrap(B, H), D, S], f32>) -> !mo.tensor<[mul_no_wrap(B, H), S, S], f32>
    %attn_mat_norm = mo.mul(%attn_mat, %norm) : !mo.tensor<[mul_no_wrap(B, H), S, S], f32>
    %attn_mat_softmax = mo.reduce.softmax(%attn_mat_norm) [axis = 2] : (!mo.tensor<[mul_no_wrap(B, H), S, S], f32>) -> !mo.tensor<[mul_no_wrap(B, H), S, S], f32>

    // This breaks pattern matching
    %attn_mat_softmax_copy = mo.custom [symbol = "copy", device = #M.device_ref<"cpu", 0>] (%attn_mat_softmax): (!mo.tensor<[mul_no_wrap(B, H), S, S], f32>) -> !mo.tensor<[mul_no_wrap(B, H), S, S], f32>

    %reproj = mo.batch_matmul(%attn_mat_softmax_copy, %value_reshaped) : (!mo.tensor<[mul_no_wrap(B, H), S, S], f32>, !mo.tensor<[mul_no_wrap(B, H), S, D], f32>) -> !mo.tensor<[mul_no_wrap(B, H), S, D], f32>

    %final_answer = mo.static.reshape(%reproj) : !mo.tensor<[mul_no_wrap(B, H), S, D], f32> -> !mo.tensor<[B, H, S, D], f32>
    mo.output %final_answer: !mo.tensor<[B, H, S, D], f32>
}

// The exact same as above, but without the fused op
// CHECK: mo.graph @cpu_flash_attn_not_pattern
// CHECK-NOT: mo.composite.no_mask_flash_attention_cpu

// CHECK-MT: --- Running 'cpu_flash_attn_not_pattern':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]
