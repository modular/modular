// COM: Check we use flash attention kernel when D <= 256
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute %t.mlir --input-shapes=1x1x6x64 --input-shapes=1x1x6x64 --input-shapes=1x1x6x64 --input-shapes="" --input-shapes=1x6x1x1 --model-inputs=random --result-output-style=full | FileCheck %s -check-prefix=CHECK-MT

// Check E2E performance. Note we also run the case where the pattern does not
// trigger (due to improper devices in graph) E2E to help establish correctness.
// Testing the pattern works on our version of llama2 produced in MIN for modcon
mo.graph @llama2_min_framework_cpu_mha<REFINE_S2, SUM_C1_S2>(
    %Q_raw: !mo.tensor<[1, REFINE_S2, 6, 64], f32>,
    %KT_raw: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>,
    %V_raw: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>,
    %norm_scalar: !mo.tensor<[], f32>,
    %attn_mask: !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>
) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32> {
    %51 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %54 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    %Q = mo.transpose(%Q_raw, %54) : (!mo.tensor<[1, REFINE_S2, 6, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32>

    // This is the K^T tensor in BHDS
    %KT = mo.transpose(%KT_raw, %51) : (!mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 6, 64, SUM_C1_S2], f32>

    // This is is BHsS, attention matrix
    // attnMatrixOp
    %attnMatrix = mo.batch_matmul(%Q, %KT) : (!mo.tensor<[1, 6, REFINE_S2, 64], f32>, !mo.tensor<[1, 6, 64, SUM_C1_S2], f32>) -> !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>

    // This is normalization BHsS
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>
    %normAttnMatrix = mo.mul(%attnMatrix, %norm) : !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>

    // Attention mask, BHsS
    %maskedNormAttnMatrix = mo.add(%normAttnMatrix, %attn_mask) : !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>

    // This is BHsS
    %softmax = mo.reduce.softmax(%maskedNormAttnMatrix) [axis = 3] : (!mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>) -> !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>

    // This output is BHsS @ BHSD --> BHsD, reprojected
    %V = mo.transpose(%V_raw, %54) : (!mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 6, SUM_C1_S2, 64], f32>
    %answer = mo.batch_matmul(%softmax, %V) : (!mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>, !mo.tensor<[1, 6, SUM_C1_S2, 64], f32>) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32>

    mo.output %answer: !mo.tensor<[1, 6, REFINE_S2, 64], f32>
}

// CHECK: mo.graph @llama2_min_framework_cpu_mha<REFINE_S2, SUM_C1_S2>(
// CHECK-SAME: [[QUERY:%[a-zA-Z0-9_]*]]: !mo.tensor<[1, REFINE_S2, 6, 64], f32>,
// CHECK-SAME: [[KEYT:%[a-zA-Z0-9_]*]]: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>,
// CHECK-SAME: [[VALUE:%[a-zA-Z0-9_]*]]: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>,
// CHECK-SAME: [[NORM:%[a-zA-Z0-9_]*]]: !mo.tensor<[], f32>,
// CHECK-SAME: [[MASK:%[a-zA-Z0-9_]*]]: !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>)
// CHECK-SAME: -> !mo.tensor<[1, 6, REFINE_S2, 64]
// CHECK-NEXT: [[V0:%.*]] = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.composite.masked_flash_attention_cpu([[QUERY]], [[KEYT]], [[VALUE]], [[MASK]], [[NORM]]) : (!mo.tensor<[1, REFINE_S2, 6, 64], f32>, !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, REFINE_S2, 6, 64], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.transpose([[V1]], [[V0]]) : (!mo.tensor<[1, REFINE_S2, 6, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[1, 6, REFINE_S2, 64], f32>

// Golden output generated from disabling the pattern.
// Regex `[0-9\-\.e ,]+` checks against fp strings like
// "-9.999551e-01, -8.299351e-01, 2.027053e-01 ..."
// CHECK-MT: --- Running 'llama2_min_framework_cpu_mha':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT:[0-9\-\.e ,]+]]
// CHECK-MT: ]

mo.graph @llama2_min_framework_cpu_mha_gt<REFINE_S2, SUM_C1_S2>(%arg0: !mo.tensor<[1, REFINE_S2, 6, 64], f32>, %arg1: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, %arg2: !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32> {
    %0 = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %1 = mo.composite.masked_flash_attention_cpu(%arg0, %arg1, %arg2, %arg4, %arg3) : (!mo.tensor<[1, REFINE_S2, 6, 64], f32>, !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[1, SUM_C1_S2, 6, 64], f32>, !mo.tensor<[1, 6, REFINE_S2, SUM_C1_S2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, REFINE_S2, 6, 64], f32>
    %2 = mo.transpose(%1, %0) : (!mo.tensor<[1, REFINE_S2, 6, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 6, REFINE_S2, 64], f32>
    mo.output %2 : !mo.tensor<[1, 6, REFINE_S2, 64], f32>
}

// CHECK-MT: --- Running 'llama2_min_framework_cpu_mha_gt':
// CHECK-MT: [
// CHECK-MT-SAME: [[RESULT]]
// CHECK-MT: ]

// COM: We cannot textually compare it with a run of fused attention because
// COM: the results are slightly different. We count on kernel tests for coverage.
