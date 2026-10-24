// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @softmax_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32> {
  %1 = rmo.mo.reduce.softmax<() -> D4, D5>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @softmax_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1 {{op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.reduce.softmax(%arg0) [axis = 1] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %1 : !mo.tensor<[D0, D1], f16>
}
