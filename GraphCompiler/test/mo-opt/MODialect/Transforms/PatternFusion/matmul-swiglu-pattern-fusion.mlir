// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s


module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "">}} {

// A gated MLP's `matmul -> split -> silu -> mul` chain is replaced by one
// `mo.composite.matmul_swiglu` and a permutation of the weight's gate/up
// halves into the adjacent rows the SM100 epilogue pairs.
// CHECK-LABEL: mo.graph @matmul_swiglu
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.split
// CHECK-NOT: mo.silu
// CHECK-NOT: mo.mul
// CHECK: %[[W:.*]] = mo.constant.external
// CHECK: %[[HALVES:.*]] = mo.static.reshape(%[[W]]) : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}> -> !mo.tensor<[2, 256, 256], bf16, gpu:0>
// CHECK: %[[PAIRS:.*]] = mo.transpose(%[[HALVES]], %{{.*}}) : (!mo.tensor<[2, 256, 256], bf16, gpu:0>, !mo.tensor<[3], si64>) -> !mo.tensor<[256, 2, 256], bf16, gpu:0>
// CHECK: %[[PERMUTED:.*]] = mo.static.reshape(%[[PAIRS]]) : !mo.tensor<[256, 2, 256], bf16, gpu:0> -> !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
// CHECK: %[[OUT:.*]] = mo.composite.matmul_swiglu(%{{.*}}, %[[PERMUTED]]) : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 256], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>) -> !mo.tensor<[64, 256], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[64, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[64, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[64, 256], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[64, 256], bf16, gpu:0>
}

// A layout rebind of an external weight remains a constant-built weight.
// CHECK-LABEL: mo.graph @matmul_swiglu_rebound_weight
// CHECK: %[[OUT:.*]] = mo.composite.matmul_swiglu(
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_rebound_weight(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0>
  %2 = mo.rebind(%1) : !mo.tensor<[16, 8], bf16, gpu:0> -> !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %3 = mo.matmul(%arg0, %2) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %4:2 = mo.split(%3, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %5 = mo.silu(%4#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %6 = mo.mul(%5, %4#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %6 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// A deep expression of non-aliased constants can be permuted during init.
// CHECK-LABEL: mo.graph @matmul_swiglu_constant_expression
// CHECK: %[[OUT:.*]] = mo.composite.matmul_swiglu(
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_constant_expression(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.add(%1, %1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %3 = mo.add(%2, %1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %4 = mo.add(%3, %1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %5 = mo.add(%4, %1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %6 = mo.add(%5, %1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %7 = mo.matmul(%arg0, %6) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %8:2 = mo.split(%7, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %9 = mo.silu(%8#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %10 = mo.mul(%9, %8#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %10 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// An expression that also depends on a runtime weight is not init-only.
// CHECK-LABEL: mo.graph @neg_mixed_weight_expression
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_mixed_weight_expression(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>, %arg1: !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.add(%1, %arg1) : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %3 = mo.matmul(%arg0, %2) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %4:2 = mo.split(%3, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %5 = mo.silu(%4#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %6 = mo.mul(%5, %4#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %6 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// An aliased weight can change between invocations and must not be permuted.
// CHECK-LABEL: mo.graph @neg_aliased_weight
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_aliased_weight(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>, hasAlias = true] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// A `[N]` bias added to the matmul result — a `bias=True` Linear — selects the
// bias variant, and the bias is permuted like the weight.
// CHECK-LABEL: mo.graph @matmul_swiglu_bias
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.split
// CHECK-NOT: mo.silu
// CHECK-NOT: mo.mul
// CHECK-NOT: mo.add
// CHECK-NOT: mo.static.broadcast_to
// CHECK: %[[BIAS_HALVES:.*]] = mo.constant {{.*}}: !mo.tensor<[2, 8], bf16, gpu:0>
// CHECK: %[[BIAS_PAIRS:.*]] = mo.transpose(%[[BIAS_HALVES]], %{{.*}}) : (!mo.tensor<[2, 8], bf16, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 2], bf16, gpu:0>
// CHECK: %[[BIAS_PERMUTED:.*]] = mo.static.reshape(%[[BIAS_PAIRS]]) : !mo.tensor<[8, 2], bf16, gpu:0> -> !mo.tensor<[16], bf16, gpu:0>
// CHECK: %[[OUT:.*]] = mo.composite.matmul_swiglu_bias(%{{.*}}, %{{.*}}, %[[BIAS_PERMUTED]])
// CHECK-SAME: -> !mo.tensor<[8, 8], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_bias(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0> : tensor<16xbf16>> : !mo.tensor<[16], bf16, gpu:0>
  %3 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %4 = mo.static.broadcast_to(%2) : !mo.tensor<[16], bf16, gpu:0> -> !mo.tensor<[8, 16], bf16, gpu:0>
  %5 = mo.add(%3, %4) : !mo.tensor<[8, 16], bf16, gpu:0>
  %6:2 = mo.split(%5, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %7 = mo.silu(%6#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %8 = mo.mul(%7, %6#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %8 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// An aliased bias must not be permuted on every invocation either.
// CHECK-LABEL: mo.graph @neg_aliased_bias
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_aliased_bias(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.constant.external [name = "bias", align = 16, device = #M.device_ref<"gpu", 0>, hasAlias = true] : !mo.tensor<[16], bf16, gpu:0>
  %3 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %4 = mo.static.broadcast_to(%2) : !mo.tensor<[16], bf16, gpu:0> -> !mo.tensor<[8, 16], bf16, gpu:0>
  %5 = mo.add(%3, %4) : !mo.tensor<[8, 16], bf16, gpu:0>
  %6:2 = mo.split(%5, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %7 = mo.silu(%6#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %8 = mo.mul(%7, %6#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %8 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// The output row stride must be 16-byte aligned even with bias.
// CHECK-LABEL: mo.graph @neg_unaligned_h
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_unaligned_h(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[8, 4], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[8, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0> : tensor<8xbf16>> : !mo.tensor<[8], bf16, gpu:0>
  %3 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 8], bf16, gpu:0>
  %4 = mo.static.broadcast_to(%2) : !mo.tensor<[8], bf16, gpu:0> -> !mo.tensor<[8, 8], bf16, gpu:0>
  %5 = mo.add(%3, %4) : !mo.tensor<[8, 8], bf16, gpu:0>
  %6:2 = mo.split(%5, %0) [axis = 1] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 4], bf16, gpu:0>, !mo.tensor<[8, 4], bf16, gpu:0>)
  %7 = mo.silu(%6#0) : !mo.tensor<[8, 4], bf16, gpu:0>
  %8 = mo.mul(%7, %6#1) : !mo.tensor<[8, 4], bf16, gpu:0>
  mo.output %8 : !mo.tensor<[8, 4], bf16, gpu:0>
}

// The activation and weight row strides must also be 16-byte aligned.
// CHECK-LABEL: mo.graph @neg_unaligned_k
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_unaligned_k(%arg0: !mo.tensor<[8, 9], bf16, gpu:0>) -> !mo.tensor<[8, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 9], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 9], bf16, gpu:0>, !mo.tensor<[16, 9], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[8, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[8, 8], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[8, 8], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[8, 8], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[8, 8], bf16, gpu:0>
}

// The gated dimension M stays dynamic.
// CHECK-LABEL: mo.graph @matmul_swiglu_dynamic_m
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.silu
// CHECK: %[[OUT:.*]] = mo.composite.matmul_swiglu(
// CHECK-SAME: -> !mo.tensor<[dyn_m, 256], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_dynamic_m(%arg0: !mo.tensor<[dyn_m, 256], bf16, gpu:0>) -> !mo.tensor<[dyn_m, 256], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[dyn_m, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[dyn_m, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = -1] : (!mo.tensor<[dyn_m, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[dyn_m, 256], bf16, gpu:0>, !mo.tensor<[dyn_m, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[dyn_m, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[dyn_m, 256], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[dyn_m, 256], bf16, gpu:0>
}

// A block that keeps its activations at rank 3: the matmul's [M, 2H] result is
// reshaped up to the block's layout, and the fused result is reshaped back.
// CHECK-LABEL: mo.graph @matmul_swiglu_rank3
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.split
// CHECK-NOT: mo.silu
// CHECK-NOT: mo.mul
// CHECK: %[[PERMUTED:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[256, 2, 256], bf16, gpu:0> -> !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
// CHECK: %[[FUSED:.*]] = mo.composite.matmul_swiglu(%{{.*}}, %[[PERMUTED]]) : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 256], bf16, gpu:0>
// CHECK: %[[OUT:.*]] = mo.static.reshape(%[[FUSED]]) : !mo.tensor<[64, 256], bf16, gpu:0> -> !mo.tensor<[2, 32, 256], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_rank3(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>) -> !mo.tensor<[2, 32, 256], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3 = mo.static.reshape(%2) : !mo.tensor<[64, 512], bf16, gpu:0> -> !mo.tensor<[2, 32, 512], bf16, gpu:0>
  %4:2 = mo.split(%3, %0) [axis = 2] : (!mo.tensor<[2, 32, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 32, 256], bf16, gpu:0>, !mo.tensor<[2, 32, 256], bf16, gpu:0>)
  %5 = mo.silu(%4#0) : !mo.tensor<[2, 32, 256], bf16, gpu:0>
  %6 = mo.mul(%5, %4#1) : !mo.tensor<[2, 32, 256], bf16, gpu:0>
  mo.output %6 : !mo.tensor<[2, 32, 256], bf16, gpu:0>
}

// CHECK-LABEL: mo.graph @matmul_swiglu_bias_rank3
// CHECK: %[[FUSED:.*]] = mo.composite.matmul_swiglu_bias(
// CHECK-SAME: -> !mo.tensor<[8, 8], bf16, gpu:0>
// CHECK: %[[OUT:.*]] = mo.static.reshape(%[[FUSED]]) : !mo.tensor<[8, 8], bf16, gpu:0> -> !mo.tensor<[2, 4, 8], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_swiglu_bias_rank3(%arg0: !mo.tensor<[8, 8], bf16, gpu:0>) -> !mo.tensor<[2, 4, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.constant.external [name = "bias", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[16], bf16, gpu:0>
  %3 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[8, 8], bf16, gpu:0>, !mo.tensor<[16, 8], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[8, 16], bf16, gpu:0>
  %4 = mo.static.broadcast_to(%2) : !mo.tensor<[16], bf16, gpu:0> -> !mo.tensor<[8, 16], bf16, gpu:0>
  %5 = mo.add(%4, %3) : !mo.tensor<[8, 16], bf16, gpu:0>
  %6 = mo.static.reshape(%5) : !mo.tensor<[8, 16], bf16, gpu:0> -> !mo.tensor<[2, 4, 16], bf16, gpu:0>
  %7:2 = mo.split(%6, %0) [axis = 2] : (!mo.tensor<[2, 4, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4, 8], bf16, gpu:0>, !mo.tensor<[2, 4, 8], bf16, gpu:0>)
  %8 = mo.silu(%7#0) : !mo.tensor<[2, 4, 8], bf16, gpu:0>
  %9 = mo.mul(%8, %7#1) : !mo.tensor<[2, 4, 8], bf16, gpu:0>
  mo.output %9 : !mo.tensor<[2, 4, 8], bf16, gpu:0>
}

// Splitting a reshaped matmul along a leading axis does not pair columns.
// CHECK-LABEL: mo.graph @neg_split_wrong_axis
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_split_wrong_axis(%arg0: !mo.tensor<[2, 16], bf16, gpu:0>) -> !mo.tensor<[2, 16], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[32, 16], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[2, 16], bf16, gpu:0>, !mo.tensor<[32, 16], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 32], bf16, gpu:0>
  %3 = mo.static.reshape(%2) : !mo.tensor<[2, 32], bf16, gpu:0> -> !mo.tensor<[4, 16], bf16, gpu:0>
  %4:2 = mo.split(%3, %0) [axis = 0] : (!mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 16], bf16, gpu:0>, !mo.tensor<[2, 16], bf16, gpu:0>)
  %5 = mo.silu(%4#0) : !mo.tensor<[2, 16], bf16, gpu:0>
  %6 = mo.mul(%5, %4#1) : !mo.tensor<[2, 16], bf16, gpu:0>
  mo.output %6 : !mo.tensor<[2, 16], bf16, gpu:0>
}

// A last-axis split after changing the matmul's width is not a SwiGLU pair.
// CHECK-LABEL: mo.graph @neg_reshape_wrong_width
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_reshape_wrong_width(%arg0: !mo.tensor<[2, 16], bf16, gpu:0>) -> !mo.tensor<[4, 8], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[32, 16], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[2, 16], bf16, gpu:0>, !mo.tensor<[32, 16], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 32], bf16, gpu:0>
  %3 = mo.static.reshape(%2) : !mo.tensor<[2, 32], bf16, gpu:0> -> !mo.tensor<[4, 16], bf16, gpu:0>
  %4:2 = mo.split(%3, %0) [axis = 1] : (!mo.tensor<[4, 16], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[4, 8], bf16, gpu:0>, !mo.tensor<[4, 8], bf16, gpu:0>)
  %5 = mo.silu(%4#0) : !mo.tensor<[4, 8], bf16, gpu:0>
  %6 = mo.mul(%5, %4#1) : !mo.tensor<[4, 8], bf16, gpu:0>
  mo.output %6 : !mo.tensor<[4, 8], bf16, gpu:0>
}

// Negative: one split half is consumed twice, so the fused op cannot drop the
// [M, 2H] intermediate.
// CHECK-LABEL: mo.graph @neg_split_half_has_other_consumers
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_split_half_has_other_consumers(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>) {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[64, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[64, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[64, 256], bf16, gpu:0>
  mo.output %5, %3#1 : !mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>
}

// Negative: the SiLU operand also has another consumer.
// CHECK-LABEL: mo.graph @neg_silu_half_has_other_consumers
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_silu_half_has_other_consumers(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>) {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant.external [name = "gate_up", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[64, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[64, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[64, 256], bf16, gpu:0>
  mo.output %5, %3#0 : !mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>
}

// Negative: the weight is a runtime value, so permuting it would cost a kernel
// on every invocation.
// CHECK-LABEL: mo.graph @neg_weight_is_not_a_constant
// CHECK-NOT: mo.composite.matmul_swiglu
// CHECK: mo.matmul
// CHECK: mo.silu
// CHECK: mo.mul
// CHECK: mo.output
mo.graph @neg_weight_is_not_a_constant(%arg0: !mo.tensor<[64, 256], bf16, gpu:0>, %arg1: !mo.tensor<[512, 256], bf16, gpu:0>) -> !mo.tensor<[64, 256], bf16, gpu:0> {
  %0 = mo.constant <#M.dense_array<256, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.rebind(%arg1) : !mo.tensor<[512, 256], bf16, gpu:0> -> !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %2 = mo.matmul(%arg0, %1) [transpose_b = true] : (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[512, 256], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[64, 512], bf16, gpu:0>
  %3:2 = mo.split(%2, %0) [axis = 1] : (!mo.tensor<[64, 512], bf16, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[64, 256], bf16, gpu:0>, !mo.tensor<[64, 256], bf16, gpu:0>)
  %4 = mo.silu(%3#0) : !mo.tensor<[64, 256], bf16, gpu:0>
  %5 = mo.mul(%4, %3#1) : !mo.tensor<[64, 256], bf16, gpu:0>
  mo.output %5 : !mo.tensor<[64, 256], bf16, gpu:0>
}
}
