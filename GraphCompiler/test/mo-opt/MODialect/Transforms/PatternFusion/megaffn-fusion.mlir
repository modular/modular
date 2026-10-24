// RUN: mo-opt %s --mo-pattern-fusion="mega-ffn-enable=true" | FileCheck %s
// RUN: mo-opt %s --mo-pattern-fusion="mega-ffn-enable=false" | FileCheck %s --check-prefix=DISABLED

// The target arch is read from the module's `mo.device_info_mapping`. The
// MegaFFN pattern arch-gates itself to sm_100a via DeviceInfoMapping (mirroring
// MatmulAddPattern) and is additionally behind the `mega-ffn-enable` pass
// option. This module is sm_100a, so the chain fuses when the option is on.
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "">}} {

// Positive: the gate-up (swiglu) -> down (block-scaled) NVFP4 MoE FFN chain,
// with both intermediates single-use and shared EP routing tensors, fuses into
// a single mo.composite.mega_ffn_nvfp4 on SM100 when the pass option is on.
//
// The fusion mints the persistent `arrival_count` scratch as a zero-initialized
// `mo.buffer.create`, sized to a static upper bound on `total_m_blocks`:
// (ceildiv(M, 8) + num_experts + 1) * ATOMIC_PAD = (66560/8 + 49 + 1) * 32 =
// 267840 ui32 slots. That buffer feeds the fused op as its arrival_count.
// L1 leaves its activation attributes at their defaults, which the fused op
// still carries explicitly for the kernel registration to bind.
// CHECK-LABEL: mo.graph @mega_ffn_fuses
// CHECK: mo.buffer.create
// CHECK-SAME: [267840], ui32
// CHECK: mo.composite.mega_ffn_nvfp4
// CHECK-SAME: activation = "silu", swiglu_alpha = #M.dense_array<0.000000e+00> : tensor<f32>, swiglu_limit = #M.dense_array<0.000000e+00> : tensor<f32>
// CHECK-NOT: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK-NOT: mo.composite.grouped_matmul_block_scaled
//
// Gate: pass option off -> the unfused two-op chain is preserved (and no
// arrival_count buffer is minted).
// DISABLED-LABEL: mo.graph @mega_ffn_fuses
// DISABLED: mo.composite.grouped_matmul_swiglu_nvfp4
// DISABLED: mo.composite.grouped_matmul_block_scaled
// DISABLED-NOT: mo.composite.mega_ffn_nvfp4
// DISABLED-NOT: mo.buffer.create
mo.graph @mega_ffn_fuses(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

// Positive (clamped SwiGLU / swigluoai): the fused kernel supports clamping, so
// the chain still fuses. The fused op carries L1's activation and its alpha and
// limit forward; non-canonical values make a dropped or defaulted one visible.
//
// CHECK-LABEL: mo.graph @mega_ffn_fuses_clamp
// CHECK: mo.composite.mega_ffn_nvfp4
// CHECK-SAME: activation = "swigluoai", swiglu_alpha = #M.dense_array<1.500000e+00> : tensor<f32>, swiglu_limit = #M.dense_array<5.000000e+00> : tensor<f32>
// CHECK-NOT: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK-NOT: mo.composite.grouped_matmul_block_scaled
mo.graph @mega_ffn_fuses_clamp(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "swigluoai", swiglu_alpha = #M.dense_array<1.5> : tensor<f32>, swiglu_limit = #M.dense_array<5.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

// Negative: the packed-activation intermediate (L1#0) is also returned, so it
// has more than one use. Re-materializing it under fusion is disallowed -> bail.
//
// CHECK-LABEL: mo.graph @neg_multi_use_intermediate
// CHECK: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK: mo.composite.grouped_matmul_block_scaled
// CHECK-NOT: mo.composite.mega_ffn_nvfp4
mo.graph @neg_multi_use_intermediate(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> (!mo.tensor<[66560, 7168], bf16, gpu:0>,
      !mo.tensor<[66560, 1024], ui8, gpu:0>) {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out, %down_in : !mo.tensor<[66560, 7168], bf16, gpu:0>,
      !mo.tensor<[66560, 1024], ui8, gpu:0>
}

// Negative: the down op is driven by a different expert_ids tensor than the
// gate-up op, so the two legs do not share EP routing -> bail.
//
// CHECK-LABEL: mo.graph @neg_mismatched_routing
// CHECK: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK: mo.composite.grouped_matmul_block_scaled
// CHECK-NOT: mo.composite.mega_ffn_nvfp4
mo.graph @neg_mismatched_routing(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %expert_ids2: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids2,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

// Negative: the down weight has a static leading dim (num_experts) of 129,
// above the scheduler warp's 128 expert slots -> bail.
//
// CHECK-LABEL: mo.graph @neg_too_many_experts
// CHECK: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK: mo.composite.grouped_matmul_block_scaled
// CHECK-NOT: mo.composite.mega_ffn_nvfp4
mo.graph @neg_too_many_experts(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[129, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[129, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[130], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[129], si32, gpu:0>,
    %scale_off: !mo.tensor<[129], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[129], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[129], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[129, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[129, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[129], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[129, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[129, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[130], ui32, gpu:0>, !mo.tensor<[129], si32, gpu:0>,
      !mo.tensor<[129], ui32, gpu:0>, !mo.tensor<[129], f32, gpu:0>,
      !mo.tensor<[129], f32, gpu:0>, !mo.tensor<[129], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[129, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[129, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[130], ui32, gpu:0>, !mo.tensor<[129], si32, gpu:0>,
      !mo.tensor<[129], ui32, gpu:0>, !mo.tensor<[129], f32, gpu:0>,
      !mo.tensor<[129], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}


// Positive (per-row input scales on the gate-up leg): the fused op takes L1's
// `a_row_scales` operand and flag, and the down leg keeps its expert scales.
//
// CHECK-LABEL: mo.graph @mega_ffn_fuses_row_scales
// CHECK: mo.composite.mega_ffn_nvfp4
// CHECK-SAME: has_gate_up_a_row_scales = true
// CHECK-SAME: !mo.tensor<[66560], bf16, gpu:0>
// CHECK-NOT: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK-NOT: mo.composite.grouped_matmul_block_scaled
mo.graph @mega_ffn_fuses_row_scales(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %row_scales: !mo.tensor<[66560], bf16, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_row_scales: !mo.tensor<[66560], bf16, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %row_scales, %c_in_scales, %est_m,
      %gate_up_nae)
      [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>, has_a_row_scales = true] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[66560], bf16, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae)
 : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

// Negative: the down leg carries per-row input scales, which the fused kernel
// does not apply -> bail.
//
// CHECK-LABEL: mo.graph @neg_down_row_scales
// CHECK: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK: mo.composite.grouped_matmul_block_scaled
// CHECK-SAME: has_a_row_scales = true
// CHECK-NOT: mo.composite.mega_ffn_nvfp4
mo.graph @neg_down_row_scales(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[50], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[49], si32, gpu:0>,
    %scale_off: !mo.tensor<[49], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[49], f32, gpu:0>,
    %row_scales: !mo.tensor<[66560], bf16, gpu:0>,
    %c_in_scales: !mo.tensor<[49], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[49], f32, gpu:0>,
    %down_row_scales: !mo.tensor<[66560], bf16, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[49, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[49], f32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_row_scales, %est_m, %down_nae)
      [has_a_row_scales = true] : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[49, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[49, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[50], ui32, gpu:0>, !mo.tensor<[49], si32, gpu:0>,
      !mo.tensor<[49], ui32, gpu:0>, !mo.tensor<[49], f32, gpu:0>,
      !mo.tensor<[66560], bf16, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

// Positive at the scheduler's limit: 128 experts fill all four slots of every
// lane. The arrival buffer is (66560/8 + 128 + 1) * 32 = 270368 ui32 slots.
// CHECK-LABEL: mo.graph @mega_ffn_fuses_128_experts
// CHECK: mo.buffer.create
// CHECK-SAME: [270368], ui32
// CHECK: mo.composite.mega_ffn_nvfp4
// CHECK-NOT: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK-NOT: mo.composite.grouped_matmul_block_scaled
mo.graph @mega_ffn_fuses_128_experts(
    %hidden: !mo.tensor<[66560, 3584], ui8, gpu:0>,
    %gate_up_w: !mo.tensor<[128, 4096, 3584], ui8, gpu:0>,
    %a_scales: !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %gate_up_bs: !mo.tensor<[128, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
    %row_off: !mo.tensor<[129], ui32, gpu:0>,
    %expert_ids: !mo.tensor<[128], si32, gpu:0>,
    %scale_off: !mo.tensor<[128], ui32, gpu:0>,
    %gate_up_es: !mo.tensor<[128], f32, gpu:0>,
    %c_in_scales: !mo.tensor<[128], f32, gpu:0>,
    %est_m: !mo.tensor<[], ui32>,
    %gate_up_nae: !mo.tensor<[], ui32>,
    %down_w: !mo.tensor<[128, 7168, 1024], ui8, gpu:0>,
    %down_bs: !mo.tensor<[128, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
    %down_es: !mo.tensor<[128], f32, gpu:0>,
    %down_nae: !mo.tensor<[], ui32>
) -> !mo.tensor<[66560, 7168], bf16, gpu:0> {
  %down_in, %silu_scales = mo.composite.grouped_matmul_swiglu_nvfp4(
      %hidden, %gate_up_w, %a_scales, %gate_up_bs, %row_off, %expert_ids,
      %scale_off, %gate_up_es, %gate_up_es, %c_in_scales, %est_m,
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.0> : tensor<f32>, swiglu_limit = #M.dense_array<0.0> : tensor<f32>] : (
      !mo.tensor<[66560, 3584], ui8, gpu:0>,
      !mo.tensor<[128, 4096, 3584], ui8, gpu:0>,
      !mo.tensor<[569, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[128, 32, 112, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[129], ui32, gpu:0>, !mo.tensor<[128], si32, gpu:0>,
      !mo.tensor<[128], ui32, gpu:0>, !mo.tensor<[128], f32, gpu:0>,
      !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[128], f32, gpu:0>,
      !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> (!mo.tensor<[66560, 1024], ui8, gpu:0>,
          !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>)
  %out = mo.composite.grouped_matmul_block_scaled(
      %down_in, %down_w, %silu_scales, %down_bs, %row_off, %expert_ids,
      %scale_off, %down_es, %down_es, %est_m, %down_nae) : (
      !mo.tensor<[66560, 1024], ui8, gpu:0>,
      !mo.tensor<[128, 7168, 1024], ui8, gpu:0>,
      !mo.tensor<[569, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[128, 56, 32, 32, 4, 4], f8e4m3fn, gpu:0>,
      !mo.tensor<[129], ui32, gpu:0>, !mo.tensor<[128], si32, gpu:0>,
      !mo.tensor<[128], ui32, gpu:0>, !mo.tensor<[128], f32, gpu:0>,
      !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>
    ) -> !mo.tensor<[66560, 7168], bf16, gpu:0>
  mo.output %out : !mo.tensor<[66560, 7168], bf16, gpu:0>
}

}
