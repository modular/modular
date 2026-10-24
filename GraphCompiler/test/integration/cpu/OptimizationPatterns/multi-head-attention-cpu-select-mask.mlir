// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s --enable-var-scope

// RUN: mo-opt %s -gc-pipeline -o %t.mlir
// RUN: mt --execute %t.mlir --input-shapes=1x2x2x3 --input-shapes=1x2x2x3 --input-shapes=1x2x2x3 --input-shapes="" --input-shapes=1x6x1x1 --model-inputs=random --result-output-style=full | FileCheck %s -check-prefix=CHECK-MT

// Check E2E performance. Note we also run the case where the pattern does not
// trigger (due to improper devices in graph) E2E to help establish correctness.
// Testing the pattern works on our version of llama2 produced in MIN for modcon
mo.graph @select_mask<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>
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

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedNormAttnMatrix) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// CHECK-LABEL: mo.graph @select_mask<SEQ, FIN_SEQ, DIM>
// CHECK-SAME: ([[VARG0:%.*]]: !mo.tensor<[1, SEQ, 2, DIM], f32>, [[VARG1:%.*]]: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, [[VARG2:%.*]]: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, [[VARG3:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32> {
// CHECK-NEXT: [[V0:%.*]] = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.constant <#M.dense_array<-3.000000e+38> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.constant <#M.dense_array<false, true, true, false> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
// CHECK-NEXT: [[V3:%.*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[1, 1, 2, 2], bool> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>
// CHECK-NEXT: [[V4:%.*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V5:%.*]] = mo.not([[V3]]) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>
// CHECK-NEXT: [[V6:%.*]] = mo.cast([[V5]]) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V7:%.*]] = mo.mul([[V4]], [[V6]]) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V8:%.*]] = mo.composite.masked_flash_attention_cpu([[VARG0]], [[VARG1]], [[VARG2]], [[V7]], [[VARG3]]) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, SEQ, 2, DIM], f32>
// CHECK-NEXT: [[V9:%.*]] = mo.transpose([[V8]], [[V0]]) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>
// CHECK-NEXT: mo.output [[V9]] : !mo.tensor<[1, 2, SEQ, DIM], f32>

// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK-MT: --- Running 'select_mask':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT:[0-9\-\.e ,]+]]
// CHECK-MT: ]

// COM: as @select_mask but the select conditional is reversed and the placement
// COM: in the select is too
mo.graph @select_mask_reverse<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[1, 2, SEQ, DIM], f32> {
    %perm1 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %select_cond = mo.constant <#M.dense_array<true, false, false, true> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
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
    %maskedNormAttnMatrix = mo.select(%select_cond_bcast, %neg_inf_bcast, %normAttnMatrix) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedNormAttnMatrix) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK-MT: --- Running 'select_mask_reverse':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]

// CHECK-LABEL: mo.graph @select_mask_reverse<SEQ, FIN_SEQ, DIM>
// CHECK-SAME: ([[VARG0:%.*]]: !mo.tensor<[1, SEQ, 2, DIM], f32>, [[VARG1:%.*]]: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, [[VARG2:%.*]]: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, [[VARG3:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32> {
// CHECK-NEXT: [[V0:%.*]] = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.constant <#M.dense_array<-3.000000e+38> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.constant <#M.dense_array<true, false, false, true> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
// CHECK-NEXT: [[V3:%.*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[1, 1, 2, 2], bool> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>
// CHECK-NEXT: [[V4:%.*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V5:%.*]] = mo.cast([[V3]]) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V6:%.*]] = mo.mul([[V4]], [[V5]]) : !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>
// CHECK-NEXT: [[V7:%.*]] = mo.composite.masked_flash_attention_cpu([[VARG0]], [[VARG1]], [[VARG2]], [[V6]], [[VARG3]]) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, SEQ, 2, DIM], f32>
// CHECK-NEXT: [[V8:%.*]] = mo.transpose([[V7]], [[V0]]) : (!mo.tensor<[1, SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>
// CHECK-NEXT: mo.output [[V8]] : !mo.tensor<[1, 2, SEQ, DIM], f32>


// COM: Pattern fails to trigger due to improper device ("cpu2")
mo.graph @select_mask_no_trigger<SEQ, FIN_SEQ, DIM>(
    %Q_raw: !mo.tensor<[1, SEQ, 2, DIM], f32>,
    %KT_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %V_raw: !mo.tensor<[1, FIN_SEQ, 2, DIM], f32>,
    %norm_scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[1, 2, SEQ, DIM], f32>
{
    %perm1 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm2 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %select_cond = mo.constant <#M.dense_array<true, false, false, true> : tensor<4xsi1>> : !mo.tensor<[1, 1, 2, 2], bool>
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
    %maskedNormAttnMatrix = mo.select(%select_cond_bcast, %neg_inf_bcast, %normAttnMatrix) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], bool>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedNormAttnMatrix) [axis = 3] : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This breaks pattern matching
    %softmax_copy = mo.custom [symbol = "copy", device = #M.device_ref<"cpu", 0>] (%softmax): (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>) -> !mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %perm2) : (!mo.tensor<[1, FIN_SEQ, 2, DIM], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>
    %answer = mo.batch_matmul(%softmax_copy, %V) : (!mo.tensor<[1, 2, SEQ, FIN_SEQ], f32>, !mo.tensor<[1, 2, FIN_SEQ, DIM], f32>) -> !mo.tensor<[1, 2, SEQ, DIM], f32>

    mo.output %answer: !mo.tensor<[1, 2, SEQ, DIM], f32>
}
// The exact same as above, but without the fused op
// CHECK: mo.graph @select_mask_no_trigger
// CHECK-NOT: mo.composite.masked_flash_attention_cpu

// CHECK-MT: --- Running 'select_mask_no_trigger':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]
