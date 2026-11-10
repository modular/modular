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

// The rebind keeps canonicalization from merging the two transposes. Only the
// outer one folds into transpose_b; folding the inner one too computes
// input @ y^T instead of input @ y.
// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_rebind_transpose_cpu
// CHECK: %[[T:.*]] = mo.transpose(
// CHECK: %[[R:.*]] = mo.rebind(%[[T]])
// CHECK: mo.batch_matmul(%{{.*}}, %[[R]]) [transpose_b = true]
mo.graph @batch_matmul_transpose_b_rebind_transpose_cpu<seq>(%input: !mo.tensor<[10, 100, 1000], f32>, %y: !mo.tensor<[10, 1000, seq], f32>) -> !mo.tensor<[10, 100, 10000], f32> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %inner = mo.transpose(%y, %perm): (!mo.tensor<[10, 1000, seq], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, seq, 1000], f32>
  %rebind = mo.rebind(%inner) : !mo.tensor<[10, seq, 1000], f32> -> !mo.tensor<[10, 10000, 1000], f32>
  %outer = mo.transpose(%rebind, %perm): (!mo.tensor<[10, 10000, 1000], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, 1000, 10000], f32>
  %res = mo.batch_matmul(%input, %outer): (!mo.tensor<[10, 100, 1000], f32>, !mo.tensor<[10, 1000, 10000], f32>) -> !mo.tensor<[10, 100, 10000], f32>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32>
}

// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_cpu
// CHECK-SAME: %[[WEIGHT:[^:]+]]: !mo.tensor<[10, 10000, 1000], f32>
// CHECK: mo.batch_matmul(%{{.*}}, %[[WEIGHT]]) [transpose_b = true]
mo.graph @batch_matmul_transpose_b_cpu(%input: !mo.tensor<[10, 100, 1000], f32>, %weight: !mo.tensor<[10, 10000, 1000], f32>) -> !mo.tensor<[10, 100, 10000], f32> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %transpose_weight = mo.transpose(%weight, %perm): (!mo.tensor<[10, 10000, 1000], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, 1000, 10000], f32>
  %res = mo.batch_matmul(%input, %transpose_weight): (!mo.tensor<[10, 100, 1000], f32>, !mo.tensor<[10, 1000, 10000], f32>) -> !mo.tensor<[10, 100, 10000], f32>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32>
}

// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_gpu
// CHECK-SAME: %[[KEY:[^:]+]]: !mo.tensor<[10, seq, 1000], f32, gpu:0>
// CHECK: mo.batch_matmul(%{{.*}}, %[[KEY]]) [transpose_b = true]
mo.graph @batch_matmul_transpose_b_gpu<seq>(%input: !mo.tensor<[10, 100, 1000], f32, gpu:0>, %key: !mo.tensor<[10, seq, 1000], f32, gpu:0>) -> !mo.tensor<[10, 100, seq], f32, gpu:0> {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %transpose_key = mo.transpose(%key, %perm): (!mo.tensor<[10, seq, 1000], f32, gpu:0>, !mo.tensor<[3], si32>) -> !mo.tensor<[10, 1000, seq], f32, gpu:0>
  %res = mo.batch_matmul(%input, %transpose_key): (!mo.tensor<[10, 100, 1000], f32, gpu:0>, !mo.tensor<[10, 1000, seq], f32, gpu:0>) -> !mo.tensor<[10, 100, seq], f32, gpu:0>
  mo.output %res : !mo.tensor<[10, 100, seq], f32, gpu:0>
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
