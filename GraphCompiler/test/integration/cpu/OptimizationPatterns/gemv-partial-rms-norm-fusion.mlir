// COM: Test GEMV + Partial RMS Norm fusion pattern.
// RUN: mo-opt %s --mo-infer-layouts --mo-pattern-fusion | FileCheck %s

// ============================================================
// Positive tests - pattern should fire
// ============================================================

// Basic GEMV + partial RMS norm fusion on GPU.
// CHECK-LABEL: mo.graph @gemv_partial_rms_norm_basic
// CHECK: mo.composite.matmul_fused_partial_rms_norm
// CHECK-SAME: {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>}
// CHECK-NOT: mo.matmul
// CHECK-NOT: mo.slice
// CHECK-NOT: mo.reduce.rms_norm
mo.graph @gemv_partial_rms_norm_basic(
  %x: !mo.tensor<[1, 7168], bf16, gpu:0>,
  %W: !mo.tensor<[7168, 2112], bf16, gpu:0>,
  %gamma: !mo.tensor<[1536], bf16, gpu:0>
) -> (!mo.tensor<[1, 1536], bf16, gpu:0>, !mo.tensor<[1, 576], bf16, gpu:0>) {
  // GEMV: [1, 7168] x [7168, 2112] (transposed) -> [1, 2112]
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 7168], bf16, gpu:0>, !mo.tensor<[7168, 2112], bf16, gpu:0>)
    -> !mo.tensor<[1, 2112], bf16, gpu:0>

  // Slice for normed portion: [1, 2112] -> [1, 1536]
  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 1536> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 2112], bf16, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 1536], bf16, gpu:0>

  // RMS norm on the sliced portion (multiply_before_cast=true for fusion)
  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} :
    (!mo.tensor<[1, 1536], bf16, gpu:0>, !mo.tensor<[1536], bf16, gpu:0>) -> !mo.tensor<[1, 1536], bf16, gpu:0>

  // Slice for unnormed portion: [1, 2112] -> [1, 576]
  %start_unnormed = mo.constant <#M.dense_array<0, 1536> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 2112> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 2112], bf16, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 576], bf16, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, 1536], bf16, gpu:0>, !mo.tensor<[1, 576], bf16, gpu:0>
}

// ============================================================
// Negative tests - pattern should NOT fire
// ============================================================

// Wrong device (CPU) - pattern should not fire
// CHECK-LABEL: mo.graph @cpu_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @cpu_not_fused(
  %x: !mo.tensor<[1, 1024], f32, cpu:0>,
  %W: !mo.tensor<[1024, 512], f32, cpu:0>,
  %gamma: !mo.tensor<[256], f32, cpu:0>
) -> (!mo.tensor<[1, 256], f32, cpu:0>, !mo.tensor<[1, 256], f32, cpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, cpu:0>, !mo.tensor<[1024, 512], f32, cpu:0>)
    -> !mo.tensor<[1, 512], f32, cpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 512], f32, cpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, cpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 256], f32, cpu:0>, !mo.tensor<[256], f32, cpu:0>) -> !mo.tensor<[1, 256], f32, cpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, cpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, cpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, 256], f32, cpu:0>, !mo.tensor<[1, 256], f32, cpu:0>
}

// M > 1 (not GEMV) - pattern should not fire
// CHECK-LABEL: mo.graph @m_not_one_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @m_not_one_not_fused(
  %x: !mo.tensor<[4, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[4, 256], f32, gpu:0>, !mo.tensor<[4, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[4, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[4, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<4, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[4, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[4, 256], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[4, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[4, 256], f32, gpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<4, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[4, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[4, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[4, 256], f32, gpu:0>, !mo.tensor<[4, 256], f32, gpu:0>
}

// Missing transpose_b - pattern should not fire
// CHECK-LABEL: mo.graph @transpose_b_false_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @transpose_b_false_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, 256], f32, gpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>
}

// Non-unit step - pattern should not fire
// CHECK-LABEL: mo.graph @non_unit_step_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @non_unit_step_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[1, 128], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  // Step of 2 in column dimension
  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step_2 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step_2) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 128], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 128], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, 128], f32, gpu:0>

  %step_1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step_1) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, 128], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>
}

// Missing unnormed slice - pattern should not fire
// CHECK-LABEL: mo.graph @missing_unnormed_slice_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @missing_unnormed_slice_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[512], f32, gpu:0>
) -> !mo.tensor<[1, 512], f32, gpu:0> {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  // RMS norm on the entire output (no slicing)
  %normed = mo.reduce.rms_norm(%gemv, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[512], f32, gpu:0>) -> !mo.tensor<[1, 512], f32, gpu:0>

  mo.output %normed : !mo.tensor<[1, 512], f32, gpu:0>
}

// Matmul has other user - pattern should not fire
// CHECK-LABEL: mo.graph @matmul_has_other_user_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @matmul_has_other_user_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 512], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  // Third use of gemv - prevents fusion
  %direct_use = mo.add(%gemv, %gemv) : !mo.tensor<[1, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, 256], f32, gpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice, %direct_use : !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 512], f32, gpu:0>
}

// Normed slice has other user - pattern should not fire
// CHECK-LABEL: mo.graph @normed_slice_has_other_user_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @normed_slice_has_other_user_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  // Second use of normed_slice - prevents fusion
  %slice_copy = mo.add(%normed_slice, %normed_slice) : !mo.tensor<[1, 256], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, 256], f32, gpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice, %slice_copy : !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>
}

// Non-constant slice bounds - pattern should not fire
// CHECK-LABEL: mo.graph @non_constant_slice_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @non_constant_slice_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>,
  %dynamic_stop: !mo.tensor<[2], si64>
) -> (!mo.tensor<[1, ?], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // Dynamic stop value prevents fusion
  %normed_slice = mo.slice(%gemv, %start_normed, %dynamic_stop, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, ?], f32, gpu:0>

  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, ?], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, ?], f32, gpu:0>

  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, ?], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>
}

// multiply_before_cast=false (Llama-style) - pattern should not fire
// CHECK-LABEL: mo.graph @multiply_before_cast_false_not_fused
// CHECK: mo.reduce.rms_norm
// CHECK-NOT: matmul_fused_partial_rms_norm
mo.graph @multiply_before_cast_false_not_fused(
  %x: !mo.tensor<[1, 1024], f32, gpu:0>,
  %W: !mo.tensor<[1024, 512], f32, gpu:0>,
  %gamma: !mo.tensor<[256], f32, gpu:0>
) -> (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>) {
  %gemv = mo.matmul(%x, %W) [transpose_b = true] :
    (!mo.tensor<[1, 1024], f32, gpu:0>, !mo.tensor<[1024, 512], f32, gpu:0>)
    -> !mo.tensor<[1, 512], f32, gpu:0>

  %start_normed = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_normed = mo.constant <#M.dense_array<1, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed_slice = mo.slice(%gemv, %start_normed, %stop_normed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  // multiply_before_cast=false prevents fusion (kernel hardcodes true)
  %normed = mo.reduce.rms_norm(%normed_slice, %gamma)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>) -> !mo.tensor<[1, 256], f32, gpu:0>

  %start_unnormed = mo.constant <#M.dense_array<0, 256> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_unnormed = mo.constant <#M.dense_array<1, 512> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %unnormed_slice = mo.slice(%gemv, %start_unnormed, %stop_unnormed, %step) :
    (!mo.tensor<[1, 512], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[1, 256], f32, gpu:0>

  mo.output %normed, %unnormed_slice : !mo.tensor<[1, 256], f32, gpu:0>, !mo.tensor<[1, 256], f32, gpu:0>
}
