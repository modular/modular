// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @squeeze_shape_ok<D0, D1 -> D2>(%shape: !mo.tensor<[D0], si64>, %indices: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
  %1 = rmo.mo.squeeze_shape<() -> D3>(%shape, %indices) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>
  mo.output<D3> %1: !mo.tensor<[D3], si64>
}

// -----

mo.graph @squeeze_shape_bad_indices<D0, D1, D2>(%shape: !mo.tensor<[D0], si64>, %indices: !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<[D1, D2], f32>'}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D0], si64>
  mo.output %1: !mo.tensor<[D0], si64>
}

// -----

mo.graph @squeeze_shape_bad_result<D0, D1>(%shape: !mo.tensor<[D0], si64>, %indices: !mo.tensor<[D1], si64>) -> !mo.tensor<[D0], si32> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op failed to verify that all of {input_shape, result} have same dtype}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D0], si32>
  mo.output %1: !mo.tensor<[D0], si32>
}
