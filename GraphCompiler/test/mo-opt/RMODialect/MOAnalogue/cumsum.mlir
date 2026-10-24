// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @cumsum_ok<D0, D1>(%input: !mo.tensor<[D0, D1], f32>) -> (!mo.tensor<[D0, D1], f32>) {
  %0 = rmo.mo.cumsum(%input) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @cumsum_bad_output<D0, D1>(%input: !mo.tensor<[D0, D1], f32>) -> (!mo.tensor<[D0, D1], f16>) {
  // expected-error @+1 {{'rmo.mo.cumsum' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.cumsum(%input) [axis = 0, exclusive = 0, reverse = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}
