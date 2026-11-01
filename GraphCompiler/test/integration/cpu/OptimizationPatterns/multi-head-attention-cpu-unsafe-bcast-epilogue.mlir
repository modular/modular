// COM: Test pattern matching on a pattern with broadcasts between the softmax and the BMM
// RUN: mo-opt %s --mo-pattern-fusion="mha-allow-unsafe-bcast=0" | FileCheck %s
// RUN: mo-opt %s --mo-pattern-fusion="mha-allow-unsafe-bcast=1" | FileCheck %s --check-prefix=CHECK-ALLOW-BCAST

// RUN: mo-opt %s --gc-pipeline -o %t.mlir

// COM: Finally check runtime assertions work
// RUN: mt --execute %t.mlir --input-shapes=1x1x6x8 --input-shapes=1x1x6x8 --input-shapes=1x1x6x8 --input-shapes="" --input-shapes=1x6x1x1 --input-shapes=1 --model-inputs=ones | FileCheck %s --check-prefix=CHECK-MT
// CHECK-MT: 'unsafe_bcast_epilogue' returned tensor

// COM: "|| true" is there because mt --execute will return an error code (like it should), note it doesn't force FileCheck to always work though
// RUN: (mt --execute %t.mlir --input-shapes=1x1x6x8 --input-shapes=1x1x6x8 --input-shapes=10x1x6x8 --input-shapes="" --input-shapes=10x6x1x1 --input-shapes=10 --model-inputs=ones || true) | FileCheck %s --check-prefix=CHECK-MT-WRONG
// CHECK-MT-WRONG: 'unsafe_bcast_epilogue' returned <error:
// CHECK-MT-WRONG: MHA Pattern Detected unsafe broadcast. This is of great interest so please report!>

mo.graph @unsafe_bcast_epilogue<B, S, KV_S, H, D, BCAST1>(
    %Q_bshd: !mo.tensor<[B, S, H, D], f32>,
    %K_bshd: !mo.tensor<[B, KV_S, H, D], f32>,
    %V_bshd: !mo.tensor<[B, KV_S, H, D], f32>,
    %norm_scalar: !mo.tensor<[], f32>,
    %attn_mask: !mo.tensor<[B, H, S, KV_S], f32>,
    %extra: !mo.tensor<[BCAST1], f32> // used to feed in extra params
) -> !mo.tensor<[B, S, H, D], f32> {
    %perm0213 = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %perm0231 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %Q_bhsd = mo.transpose(%Q_bshd, %perm0213) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
    %K_bhsdT = mo.transpose(%K_bshd, %perm0231) : (!mo.tensor<[B, KV_S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, D, KV_S], f32>
    %V_bhsd = mo.transpose(%V_bshd, %perm0213) : (!mo.tensor<[B, KV_S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, KV_S, D], f32>
    %attnMatrix = mo.batch_matmul(%Q_bhsd, %K_bhsdT) : (!mo.tensor<[B, H, S, D], f32>, !mo.tensor<[B, H, D, KV_S], f32>) -> !mo.tensor<[B, H, S, KV_S], f32>
    %norm = mo.static.broadcast_to(%norm_scalar): !mo.tensor<[], f32> -> !mo.tensor<[B, H, S, KV_S], f32>
    %attnMatrixNorm = mo.mul(%attnMatrix, %norm) : !mo.tensor<[B, H, S, KV_S], f32>
    %attnMatrixNormMasked = mo.add(%attnMatrixNorm, %attn_mask) : !mo.tensor<[B, H, S, KV_S], f32>
    %unsafeBCast1 = mo.static.broadcast_to(%attnMatrixNormMasked): !mo.tensor<[B, H, S, KV_S], f32> -> !mo.tensor<[BCAST1, H, S, KV_S], f32>
    %attnMatrixSoftmax = mo.reduce.softmax(%unsafeBCast1) [axis = 3] : (!mo.tensor<[BCAST1, H, S, KV_S], f32>) -> !mo.tensor<[BCAST1, H, S, KV_S], f32>
    %unsafeBCast2 = mo.static.broadcast_to(%attnMatrixSoftmax): !mo.tensor<[BCAST1, H, S, KV_S], f32> -> !mo.tensor<[B, H, S, KV_S], f32>
    %out_bhsd = mo.batch_matmul(%unsafeBCast2, %V_bhsd) : (!mo.tensor<[B, H, S, KV_S], f32>, !mo.tensor<[B, H, KV_S, D], f32>) -> !mo.tensor<[B, H, S, D], f32>
    %out_bshd = mo.transpose(%out_bhsd, %perm0213) : (!mo.tensor<[B, H, S, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, S, H, D], f32>
    mo.output %out_bshd: !mo.tensor<[B, S, H, D], f32>
}

// COM: No replacement with the flag
// CHECK-NOT: mo.custom

// COM: With the flag, the pattern should be replaced
// CHECK-ALLOW-BCAST-LABEL: mo.graph
// CHECK-ALLOW-BCAST-SAME: @unsafe_bcast_epilogue<B, S, KV_S, H, D, BCAST1>([[VARG0:[^ ]*]]: !mo.tensor<[B, S, H, D], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[B, KV_S, H, D], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[B, KV_S, H, D], f32>, [[VARG3:[^ ]*]]: !mo.tensor<[], f32>, [[VARG4:[^ ]*]]: !mo.tensor<[B, H, S, KV_S], f32>, [[VARG5:[^ ]*]]: !mo.tensor<[BCAST1], f32>) -> !mo.tensor<[B, S, H, D], f32> {
// CHECK-ALLOW-BCAST-NEXT: kgen.param.declare MHA_BCAST_CHECK1: scalar<bool> = <identical(:scalar<si64> B, BCAST1)>
// CHECK-ALLOW-BCAST-NEXT: kgen.param.declare MHA_BCAST_CHECK2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "MHA Pattern Detected unsafe broadcast. This is of great interest so please report!" : !kgen.string}, MHA_BCAST_CHECK1)>
// CHECK-ALLOW-BCAST-NEXT: kgen.param.declare MHA_BCAST_CHECK3: scalar<bool>  = <identical(:scalar<si64> B, BCAST1)>
// CHECK-ALLOW-BCAST-NEXT: kgen.param.declare MHA_BCAST_CHECK4: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "MHA Pattern Detected unsafe broadcast. This is of great interest so please report!" : !kgen.string}, MHA_BCAST_CHECK3)>
// CHECK-ALLOW-BCAST-NEXT: [[V0:[^ ]*]] = mo.composite.masked_flash_attention_cpu([[VARG0]], [[VARG1]], [[VARG2]], [[VARG4]], [[VARG3]]) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[B, KV_S, H, D], f32>, !mo.tensor<[B, KV_S, H, D], f32>, !mo.tensor<[B, H, S, KV_S], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[B, S, H, D], f32>
// CHECK-ALLOW-BCAST-NEXT: mo.output [[V0]] : !mo.tensor<[B, S, H, D], f32>
