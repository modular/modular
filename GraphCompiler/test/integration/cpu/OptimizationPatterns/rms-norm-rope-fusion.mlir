// COM: Test RMSNorm + RoPE fusion pattern.
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// ============================================================
// Positive tests — pattern should fire and produce reduce.rms_norm.RoPE
// ============================================================

// Basic RoPE on GPU, no casts.
// CHECK-LABEL: mo.graph @rope_basic
// CHECK:       mo.composite.rms_norm_rope
// CHECK-SAME:  {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>, weight_offset = #M.dense_array<0.000000e+00> : tensor<f32>, multiply_before_cast = false}
mo.graph @rope_basic<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// RoPE with commuted add operands (rotated*sin first, normed*cos second).
// CHECK-LABEL: mo.graph @rope_commuted_add
// CHECK:       mo.composite.rms_norm_rope
mo.graph @rope_commuted_add<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  // commuted: rotated_sin comes first in the add
  %result = mo.add(%rotated_sin, %normed_cos) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// RoPE with multiply_before_cast = true.
// CHECK-LABEL: mo.graph @rope_multiply_before_cast
// CHECK:       mo.composite.rms_norm_rope
mo.graph @rope_multiply_before_cast<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// "float32 RMSNorm sandwich": the bf16 activation is upcast to f32, normalized
// in f32, and downcast back to bf16 before RoPE (the idiom torch.compile fuses,
// e.g. JSC-32). The intervening f32->bf16 downcast would otherwise block the
// RoPE match; the pattern steps over it and emits an f32-in / bf16-out
// composite (the kernel decouples its output dtype), keeping weight/epsilon in
// f32 and forcing multiply_before_cast=true. The input upcast is intentionally
// left in place for downstream prologue (FusedInput) fusion to absorb.
// CHECK-LABEL: mo.graph @rope_f32_rmsnorm_sandwich
// CHECK:       mo.cast({{.*}}) : (!mo.tensor<[seq_len, 4], bf16, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
// CHECK-NOT:   mo.reduce.rms_norm
// CHECK:       mo.composite.rms_norm_rope
// CHECK-SAME:  {epsilon = {{.*}}, weight_offset = {{.*}}, multiply_before_cast = true}
mo.graph @rope_f32_rmsnorm_sandwich<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], bf16, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], bf16, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], bf16, gpu:0>
) -> !mo.tensor<[seq_len, 4], bf16, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %input_f32 = mo.cast(%input) : (!mo.tensor<[seq_len, 4], bf16, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_f32 = mo.reduce.rms_norm(%input_f32, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed = mo.cast(%normed_f32) : (!mo.tensor<[seq_len, 4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], bf16, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], bf16, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], bf16, gpu:0>, !mo.tensor<[seq_len, 2], bf16, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], bf16, gpu:0>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], bf16, gpu:0>, !mo.tensor<[seq_len, 2], bf16, gpu:0>)
    -> !mo.tensor<[seq_len, 4], bf16, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], bf16, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], bf16, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], bf16, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], bf16, gpu:0>
}

// ============================================================
// Negative tests — pattern should NOT fire
// ============================================================

// Wrong concat order: concat(x1, -x2) instead of concat(-x2, x1).
// CHECK-LABEL: mo.graph @rope_wrong_concat_order
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.composite.rms_norm_rope
mo.graph @rope_wrong_concat_order<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  // Wrong order: (x1, neg_x2) instead of (neg_x2, x1)
  %rotated = mo.concat[1](%x1, %neg_x2) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// Missing mul(normed, cos) — add uses normed directly; pattern should not fire.
// CHECK-LABEL: mo.graph @rope_missing_cos_mul
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.composite.rms_norm_rope
mo.graph @rope_missing_cos_mul<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  // No mul(normed, cos) — use normed directly in the add
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// Split along axis 0 instead of the last axis: pattern should not fire.
// CHECK-LABEL: mo.graph @rope_wrong_split_axis
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.composite.rms_norm_rope
mo.graph @rope_wrong_split_axis(
  %input:    !mo.tensor<[4, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[4, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[4, 4], f32, gpu:0>
) -> !mo.tensor<[4, 4], f32, gpu:0> {
  // Split along axis 0 (not the last axis == 1).
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 0] :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[2, 4], f32, gpu:0>
  %cat_axis = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  %rotated = mo.concat[0](%neg_x2, %x1) :
    (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
    -> !mo.tensor<[4, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[4, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[4, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[4, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[4, 4], f32, gpu:0>
}

// ============================================================
// Text (positive) vs Vision (negative)
//
// The text decoder RoPE is rotate-half: split(normed) -> negate(x2) ->
// concat(-x2, x1), so the fusion fires. The Gemma4 *vision* RoPE is
// complex-multiplication: each split half is itself sliced into real/imag
// parts (x[..., :h] / x[..., h:]) with no `negate` at all. Once
// SliceToSplitPattern rewrites those complementary slices into a `mo.split`,
// the RMSNorm output's split-half `x2` has exactly one use -- but that use is
// a `mo.split`, not a `mo.negative`. The fusion must decline (and, crucially,
// must not abort on an unchecked `cast<NegativeOp>`).
// ============================================================

// Text rotate-half RoPE: pattern fires.
// CHECK-LABEL: mo.graph @rope_text_rotate_half
// CHECK:       mo.composite.rms_norm_rope
mo.graph @rope_text_rotate_half<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// Vision RoPE shape: x2's single use is a split, not a negate. The pattern
// must decline without aborting (regression test for the unchecked-cast fix).
// CHECK-LABEL: mo.graph @rope_vision_split_user_not_negate
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.composite.rms_norm_rope
mo.graph @rope_vision_split_user_not_negate<seq_len>(
  %input:   !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:  !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight)
    {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  // x2's single use is another split (real/imag halves), NOT a negate.
  %inner_sizes = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %x_re, %x_im = mo.split(%x2, %inner_sizes) [axis = 1] :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 1], f32, gpu:0>, !mo.tensor<[seq_len, 1], f32, gpu:0>)
  %x2_back = mo.concat[1](%x_im, %x_re) :
    (!mo.tensor<[seq_len, 1], f32, gpu:0>, !mo.tensor<[seq_len, 1], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 2], f32, gpu:0>
  %result = mo.concat[1](%x1, %x2_back) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}
