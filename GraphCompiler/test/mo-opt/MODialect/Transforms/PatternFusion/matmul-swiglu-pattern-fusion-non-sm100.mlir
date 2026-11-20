// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s


// The fused kernel runs on SM100 only, so the same gated-MLP chain on an older
// SM stays as it is.
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_90a", "">}} {

// CHECK-LABEL: mo.graph @neg_non_sm100
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_non_sm100(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>) -> !mo.tensor<[64, 256], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[64, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[64, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[64, 256], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[64, 256], bf16, gpu:0>
}
}
