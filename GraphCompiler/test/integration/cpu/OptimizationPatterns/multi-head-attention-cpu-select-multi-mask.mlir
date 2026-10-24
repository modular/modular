// COM: check examples which mix select and explicit masking

// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s --enable-var-scope

// RUN: mo-opt %s -gc-pipeline -o %t.mlir
// RUN: mt --execute %t.mlir --input-shapes=1x2x2x3 --input-shapes=1x2x2x3 --input-shapes=1x2x2x3 --input-shapes="" --input-shapes=1x2x2x2 --model-inputs=random --result-output-style=full | FileCheck %s -check-prefix=CHECK-MT

// Check E2E performance. Note we also run the case where the pattern does not
// trigger (due to improper devices in graph) E2E to help establish correctness.
// Testing the pattern works on our version of llama2 produced in MIN for modcon
mo.graph @select_add_mask<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>,
    %attn_mask: !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
) -> !mo.tensor<[1, 2, SEQ, DIM], f32> {
    %perm1 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %select_cond = mo.constant <#M.dense_array<false, true, true, false> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
    %select_cond_bcast = mo.static.broadcast_to(%select_cond): !mo.tensor<[1, 1, 2, 2], bool> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>

    %neg_inf = mo.constant <#M.dense_array<-3.e38>: tensor<1xf32>> : !mo.tensor<[], f32>
    %neg_inf_bcast = mo.static.broadcast_to(%neg_inf): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    %Q = mo.transpose(%Q_raw, %perm2) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    // This is the K^T tensor in BHDS
    %KT = mo.transpose(%KT_raw, %perm1) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>

    // This is is BHsS, attention matrix
    // attnMatrixOp
    %attnMatrix = mo.batch_matmul(%Q, %KT) : (!mo.tensor<[1, 2, SEQ, DIM], f32>, !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is normalization BHsS
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %normAttnMatrix = mo.mul(%attnMatrix, %norm) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // Attention mask, BHsS
    %maskedNormAttnMatrix = mo.select(%select_cond_bcast, %normAttnMatrix, %neg_inf_bcast) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %maskedAgain = mo.add(%maskedNormAttnMatrix, %attn_mask): !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedAgain) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// CHECK-LABEL: mo.graph @select_add_mask<SEQ, FIN_SEQ, DIM>
// CHECK: mo.composite.masked_flash_attention_cpu


// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK-MT: --- Running 'select_add_mask':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT:[0-9\-\.e ,]+]]
// CHECK-MT: ]

// COM: Add and Select masks instead of Select then Add
mo.graph @add_select_mask<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>,
    %attn_mask: !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
) -> !mo.tensor<[1, 2, SEQ, DIM], f32> {
    %perm1 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %select_cond = mo.constant <#M.dense_array<false, true, true, false> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
    %select_cond_bcast = mo.static.broadcast_to(%select_cond): !mo.tensor<[1, 1, 2, 2], bool> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>

    %neg_inf = mo.constant <#M.dense_array<-3.e38>: tensor<1xf32>> : !mo.tensor<[], f32>
    %neg_inf_bcast = mo.static.broadcast_to(%neg_inf): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    %Q = mo.transpose(%Q_raw, %perm2) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    // This is the K^T tensor in BHDS
    %KT = mo.transpose(%KT_raw, %perm1) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>

    // This is is BHsS, attention matrix
    // attnMatrixOp
    %attnMatrix = mo.batch_matmul(%Q, %KT) : (!mo.tensor<[1, 2, SEQ, DIM], f32>, !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is normalization BHsS
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %normAttnMatrix = mo.mul(%attnMatrix, %norm) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // Attention mask, BHsS
    %maskedNormAttnMatrix = mo.add(%normAttnMatrix, %attn_mask): !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %maskedAgain = mo.select(%select_cond_bcast, %maskedNormAttnMatrix, %neg_inf_bcast) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedAgain) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// CHECK-LABEL: mo.graph @add_select_mask<SEQ, FIN_SEQ, DIM>
// CHECK: mo.composite.masked_flash_attention_cpu

// CHECK-MT: --- Running 'add_select_mask':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]

// COM: Pattern should not fire because device "cpu2" should be "cpu"
mo.graph @select_add_mask_no_trigger<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>,
    %attn_mask: !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
) -> !mo.tensor<[1, 2, SEQ, DIM], f32>
{
    %perm1 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %select_cond = mo.constant <#M.dense_array<false, true, true, false> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
    %select_cond_bcast = mo.static.broadcast_to(%select_cond): !mo.tensor<[1, 1, 2, 2], bool> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>

    %neg_inf = mo.constant <#M.dense_array<-3.e38>: tensor<1xf32>> : !mo.tensor<[], f32>
    %neg_inf_bcast = mo.static.broadcast_to(%neg_inf): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    %Q = mo.transpose(%Q_raw, %perm2) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    // This is the K^T tensor in BHDS
    %KT = mo.transpose(%KT_raw, %perm1) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>

    // This is is BHsS, attention matrix
    // attnMatrixOp
    %attnMatrix = mo.batch_matmul(%Q, %KT) : (!mo.tensor<[1, 2, SEQ, DIM], f32>, !mo.tensor<[1, 2, DIM, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is normalization BHsS
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %normAttnMatrix = mo.mul(%attnMatrix, %norm) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // Attention mask, BHsS
    %maskedNormAttnMatrix = mo.add(%normAttnMatrix, %attn_mask): !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
    %maskedAgain = mo.select(%select_cond_bcast, %maskedNormAttnMatrix, %neg_inf_bcast) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedAgain) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This breaks pattern matching
    %softmax_copy = mo.custom [symbol = "copy", device = #M.device_ref<"cpu", 0>] (%softmax): (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax_copy, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// The exact same as above, but without the fused op
// CHECK: mo.graph @select_add_mask_no_trigger
// CHECK-NOT: mo.composite.masked_flash_attention_cpu

// CHECK-MT: --- Running 'select_add_mask_no_trigger':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]
