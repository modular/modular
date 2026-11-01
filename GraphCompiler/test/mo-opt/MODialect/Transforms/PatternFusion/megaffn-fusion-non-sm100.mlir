// RUN: mo-opt %s --mo-pattern-fusion="mega-ffn-enable=true" | FileCheck %s
// RUN: mo-opt %s --mo-pattern-fusion="mega-ffn-enable=true" | mo-opt - --mo-pattern-fusion="mega-ffn-enable=true" | FileCheck %s

// Arch negative: a non-SM100 (sm_90a) module. The MegaFFN pattern arch-gates
// itself to sm_100a via DeviceInfoMapping (mirroring MatmulAddPattern), so the
// gate-up -> down chain stays unfused even with the pass option enabled.
module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_90a", "">}} {

// CHECK-LABEL: mo.graph @neg_non_sm100
// CHECK: mo.composite.grouped_matmul_swiglu_nvfp4
// CHECK-SAME: swiglu_alpha = #M.dense_array<0.000000e+00> : tensor<f32>
// CHECK-SAME: swiglu_limit = #M.dense_array<0.000000e+00> : tensor<f32>
// CHECK: mo.composite.grouped_matmul_block_scaled
// CHECK-NOT: mo.composite.mega_ffn_nvfp4
mo.graph @neg_non_sm100(
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
      %gate_up_nae) [activation = "silu", swiglu_alpha = #M.dense_array<0.000000e+00> : tensor<f32>, swiglu_limit = #M.dense_array<0.000000e+00> : tensor<f32>] : (
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

}
