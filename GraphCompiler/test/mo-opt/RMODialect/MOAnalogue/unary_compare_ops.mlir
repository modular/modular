// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @is_nan_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}

// -----

mo.graph @is_nan_bad_result<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.is_nan' op result #0 must be boolean element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @is_inf_ok<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool> {
  %1 = rmo.mo.is_inf(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
  mo.output %1 : !mo.tensor<[D0, D1], bool>
}
