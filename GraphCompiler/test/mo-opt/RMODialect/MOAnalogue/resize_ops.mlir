// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize_linear_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.resize.linear<() -> P0, P1, P2, P3>(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = false]: (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @resize_linear_bad_operands<N, C, H, W, D0, D1>(%arg0: !mo.tensor<[N, C, H, W], f32>, %size: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[N, C, H, W], f32> {
  // expected-error @+1 {{'rmo.mo.resize.linear' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D0, D1], f32>'}}
  %0 = rmo.mo.resize.linear(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = false]: (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[N, C, H, W], f32>
  mo.output %0 : !mo.tensor<[N, C, H, W], f32>
}

// -----

mo.graph @resize_nearest_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.resize.nearest<() -> P0, P1, P2, P3>(%arg0, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>, round_mode = 0]: (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @resize_bicubic_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.resize.bicubic<() -> P0, P1, P2, P3>(%arg0, %size) : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}
