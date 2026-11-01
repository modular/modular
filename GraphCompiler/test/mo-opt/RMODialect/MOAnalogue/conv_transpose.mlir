// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @conv_transpose_ok<N, C, H, W, KC, KH, KW -> O0, O1, O2, O3>(%input : !mo.tensor<[N, C, H, W], f32>, %filter : !mo.tensor<[KC, C, KH, KW], f32>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>, %op: !mo.tensor<[2], si64>) -> (!mo.tensor<[O0, O1, O2, O3], f32>) {
  %0 = rmo.mo.conv_transpose<() -> P0, P1, P2, P3>(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[N, C, H, W], f32>, !mo.tensor<[KC, C, KH, KW], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @conv_transpose_bad_operand<N, C, H, W, KC, KH, KW, D0, D1>(%input : !mo.tensor<[N, C, H, W], f32>, %filter : !mo.tensor<[KC, C, KH, KW], f32>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[D0, D1], f32>, %pa: !mo.tensor<[4], si64>, %op: !mo.tensor<[2], si64>) -> (!mo.tensor<[N, C, H, W], f32>) {
  // expected-error@+1 {{'rmo.mo.conv_transpose' op operand #3 must be rank-1 tensor with indices, but got '!mo.tensor<[D0, D1], f32>'}}
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[N, C, H, W], f32>, !mo.tensor<[KC, C, KH, KW], f32>, !mo.tensor<[2], si64>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[N, C, H, W], f32>
  mo.output %0 : !mo.tensor<[N, C, H, W], f32>
}
