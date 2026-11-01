// RUN: mo-opt %s --split-input-file --verify-diagnostics

// A mean is a sum divided by the length of the reduced axis.
mo.graph @mean_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %len: !mo.tensor<[], f32>) -> !mo.tensor<[D2, D3], f32> {
  %sum = rmo.mo.reduce.add<() -> D4, D5>(%arg0) [axis = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  %1 = rmo.div(%sum, %len) : (!mo.tensor<[D4, D5], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @sum_bad_output<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error@+1 {{'rmo.mo.reduce.add' op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.reduce.add(%arg0) [axis = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %1 : !mo.tensor<[D0, D1], f16>
}

// -----

mo.graph @argmin_ok<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], si64> {
  %1 = rmo.mo.reduce.arg_min<() -> D4, D5>(%arg0) [axis = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], si64>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], si64>
}

// -----

mo.graph @argmin_bad_output<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.reduce.arg_min' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<[D0, D1], f32>'}}
  %1 = rmo.mo.reduce.arg_min(%arg0) [axis = 0] : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
