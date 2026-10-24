// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @gather_ok<D0, D1, I -> D2, D3>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I], si32>) -> !mo.tensor<[D2, D3], f32> {
  %0 = rmo.mo.gather<() -> D4, D5>(%input, %indices) [axis = 0] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I], si32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %0 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @gather_bad_indices<D0, D1, I>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[I], f32>'}}
  %0 = rmo.mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @gather_bad_result<D0, D1, I>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I], si32>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1 {{'rmo.mo.gather' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I], si32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}

// -----

mo.graph @gather_nd_ok<D0, D1, I0, I1 -> D2, D3>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I0, I1], si32>) -> !mo.tensor<[D2, D3], f32> {
  %0 = rmo.mo.gather_nd<() -> D4, D5>(%input, %indices) [batchDims = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I0, I1], si32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %0 : !mo.tensor<[D4, D5], f32>
}

// -----

mo.graph @gather_nd_bad_result<D0, D1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I0, I1], si32>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather_nd(%input, %indices) [batchDims = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I0, I1], si32>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}

// -----

mo.graph @gather_nd_bad_indices<D0, D1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[I0, I1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[I0, I1], f32>'}}
  %0 = rmo.mo.gather_nd(%input, %indices) [batchDims = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[I0, I1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}
