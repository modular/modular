// RUN: mo-opt %s --mo-pattern-fusion  | FileCheck %s

// CHECK: mo.graph @split_zero_axis(%[[INPUT:.*]]:
// CHECK: %[[RES:.*]]:2 = mogg.split(%[[INPUT]])
// CHECK: mo.output %[[RES]]#0, %[[RES]]#1

mo.graph @split_zero_axis(
  %input: !mo.tensor<[5, 4], f32>
) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>) {
  %split_sizes = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res:2 = mo.split(%input, %split_sizes) [axis = 0] : (
    !mo.tensor<[5, 4], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>
}

// CHECK-LABEL: mo.graph @concat
// CHECK-NEXT: %[[IN1:.*]] = mo.static.reshape
// CHECK-NEXT: %[[IN2:.*]] = mo.static.reshape
// CHECK-NEXT: %[[OUT:.*]] = mogg.concat(%[[IN1]], %[[IN2]])
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @concat(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[4, 2, 4], f32>) {
  %0 = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %1 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %3 = mo.concat[0] (%1, %2) : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[4, 2, 4], f32>)
  mo.output %3 : !mo.tensor<[4, 2, 4], f32>
}

// CHECK-LABEL: mo.graph @concat_gpu
// CHECK-NEXT: %[[IN1:.*]] = mo.static.reshape
// CHECK-NEXT: %[[IN2:.*]] = mo.static.reshape
// CHECK-NEXT: %[[OUT:.*]] = mogg.concat(%[[IN1]], %[[IN2]])
// CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[4, 2, 4], f32, gpu:0>

mo.graph @concat_gpu(%arg0: !mo.tensor<[2, 8], f32, gpu:0>) -> !mo.tensor<[4, 2, 4], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<0> : tensor<1xsi32>> : !mo.tensor<[], si32>
  %1 = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32, gpu:0> -> !mo.tensor<[2, 2, 4], f32, gpu:0>
  %2 = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32, gpu:0> -> !mo.tensor<[2, 2, 4], f32, gpu:0>
  %3 = mo.concat[0] (%1, %2) : (!mo.tensor<[2, 2, 4], f32, gpu:0>, !mo.tensor<[2, 2, 4], f32, gpu:0>) -> !mo.tensor<[4, 2, 4], f32, gpu:0>
  mo.output %3 : !mo.tensor<[4, 2, 4], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_cpu
// CHECK: mo.batch_matmul(%{{.*}}, %{{.*}}) [transpose_b = true]
mo.graph @batch_matmul_transpose_b_cpu(%input: !mo.tensor<[10, 100, 1000], f32>, %weight: !mo.tensor<[10, 10000, 1000], f32>) -> !mo.tensor<[10, 100, 10000], f32> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %transpose_weight = mo.transpose(%weight, %perm): (!mo.tensor<[10, 10000, 1000], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, 1000, 10000], f32>
  %res = mo.batch_matmul(%input, %transpose_weight): (!mo.tensor<[10, 100, 1000], f32>, !mo.tensor<[10, 1000, 10000], f32>) -> !mo.tensor<[10, 100, 10000], f32>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32>
}

// `transpose_b=True` on GPU for batch matmul is not supported rn.
// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_gpu_disabled
// CHECK-NOT: [transpose_b = true]
mo.graph @batch_matmul_transpose_b_gpu_disabled(%input: !mo.tensor<[10, 100, 1000], f32, gpu:0>, %weight: !mo.tensor<[10, 10000, 1000], f32, gpu:0>) -> !mo.tensor<[10, 100, 10000], f32, gpu:0> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %transpose_weight = mo.transpose(%weight, %perm): (!mo.tensor<[10, 10000, 1000], f32, gpu:0>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, 1000, 10000], f32, gpu:0>
  %res = mo.batch_matmul(%input, %transpose_weight): (!mo.tensor<[10, 100, 1000], f32, gpu:0>, !mo.tensor<[10, 1000, 10000], f32, gpu:0>) -> !mo.tensor<[10, 100, 10000], f32, gpu:0>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32, gpu:0>
}


// CHECK-LABEL: mo.graph @matmul_transpose_b_indirect_constant_weight
// make sure that KN layout is transpose to NK layout, without a concrete transpose operation.
// CHECK-NOT: mo.transpose
// CHECK-NOT: #mo.layout<KN>
// CHECK:     #mo.layout<NK>
mo.graph @matmul_transpose_b_indirect_constant_weight(%input: !mo.tensor<[1000, 4096], bf16, gpu:0>) -> !mo.tensor<[1000, 3584], bf16, gpu:0> {
  %perm= mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si64>
  %weight = mo.constant.external [name = "blk.31.ffn_up.weight", align = 2, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[3584, 4096], bf16>
  %ch0 = mo.chain.create()
  %gpu_weight, %ch1 = mo.transfer[%ch0] %weight : !mo.tensor<[3584, 4096], bf16> to <"gpu", 0>
  %transpose_weight = mo.transpose(%gpu_weight, %perm) : (!mo.tensor<[3584, 4096], bf16, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[4096, 3584], bf16, gpu:0, {layout = #mo.layout<KN>}>
  %res = mo.matmul(%input, %transpose_weight) : (!mo.tensor<[1000, 4096], bf16, gpu:0>, !mo.tensor<[4096, 3584], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1000, 3584], bf16, gpu:0>
  mo.output %res : !mo.tensor<[1000, 3584], bf16, gpu:0>
}


// CHECK-LABEL: mo.graph @batch_matmul_scale_on_gpu
// CHECK: %[[SCALE:.*]], %{{.*}} = mo.transfer
// CHECK: %[[RESULT:.*]] = mo.composite.masked_flash_attention_gpu(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SCALE]])
// CHECK: mo.output %[[RESULT]]
mo.graph @batch_matmul_scale_on_gpu<batch, seq_len>(
  %arg1: !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>,

  %Q: !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>,
  %K: !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>,
  %V: !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>
) -> !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0> {
    %0 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %2 = mo.constant <#M.dense_array<8.837890e-02> : tensor<bf16>> : !mo.tensor<[], bf16, gpu:0>
    %3 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>

    // This is the scale value on the GPU
    %4 = mo.static.broadcast_to(%2) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>

    %5 = mo.transpose(%Q, %3) : (!mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>
    %6 = mo.transpose(%K, %3) : (!mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>
    %7 = mo.transpose(%V, %0) : (!mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[batch, 18, 128, seq_len], bf16, gpu:0>

    %8 = mo.batch_matmul(%5, %7) : (!mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>, !mo.tensor<[batch, 18, 128, seq_len], bf16, gpu:0>) -> !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>
    %9 = mo.mul(%8, %4) : !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>
    %10 = mo.add(%9, %arg1) : !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>
    %11 = mo.reduce.softmax(%10) [axis = 3] : (!mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>) -> !mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>

    %12 = mo.batch_matmul(%11, %6) : (!mo.tensor<[batch, 18, seq_len, seq_len], bf16, gpu:0>, !mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>) -> !mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>
    %13 = mo.transpose(%12, %3) : (!mo.tensor<[batch, 18, seq_len, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>

    mo.output %13 : !mo.tensor<[batch, seq_len, 18, 128], bf16, gpu:0>
}
