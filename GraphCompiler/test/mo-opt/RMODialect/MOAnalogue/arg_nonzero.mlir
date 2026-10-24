// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @argnonzero_ok<D0, D1 -> C>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[C, 2], si64> {
  %0 = rmo.mo.arg_nonzero<() -> R>(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[R, 2], si64>
  mo.output<R> %0 : !mo.tensor<[R, 2], si64>
}

// -----

mo.graph @argnonzero_bad_output<D0, D1, C>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[C, 2], f32> {
  // expected-error @+1 {{'rmo.mo.arg_nonzero' op result #0 must be rank 2 signed 64 integer tensor, but got '!mo.tensor<[C, 2], f32>'}}
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[C, 2], f32>
  mo.output %0 : !mo.tensor<[C, 2], f32>
}

// -----

mo.graph @argnonzero_bad_rank<D0, D1, C>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[C], si64> {
  // expected-error @+1 {{'rmo.mo.arg_nonzero' op result #0 must be rank 2 signed 64 integer tensor, but got '!mo.tensor<[C], si64>'}}
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[C], si64>
  mo.output %0 : !mo.tensor<[C], si64>
}
