// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @max_pool_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.max_pool<() -> P0, P1, P2, P3>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @max_pool_bad_operand<N, C, H, W, D0, D1>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[D0, D1], f32>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[N, C, H, W], f32> {
  // expected-error @+1{{'rmo.mo.max_pool' op operand #1 must be rank-1 tensor with indices, but got '!mo.tensor<[D0, D1], f32>'}}
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[N, C, H, W], f32>
  mo.output %0 : !mo.tensor<[N, C, H, W], f32>
}

// -----

mo.graph @max_pool_bad_result<N, C, H, W>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[N, C, H, W], f16> {
  // expected-error @+1{{'rmo.mo.max_pool' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[N, C, H, W], f16>
  mo.output %0 : !mo.tensor<[N, C, H, W], f16>
}

// -----

mo.graph @max_pool_ceil_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.max_pool_ceil_mode_true<() -> P0, P1, P2, P3>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @avg_pool_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.avg_pool<() -> P0, P1, P2, P3>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}


// -----

mo.graph @avg_pool_ceil_ok<N, C, H, W -> O0, O1, O2, O3>(%arg0: !mo.tensor<[N, C, H, W], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.avg_pool_ceil_mode_true<() -> P0, P1, P2, P3>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}
