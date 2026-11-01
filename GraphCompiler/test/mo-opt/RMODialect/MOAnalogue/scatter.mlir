// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scatter_ok<D0, D1, U0, U1, I0, I1 -> R0, R1>(%input: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1], f32>, %indices: !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[R0, R1], f32> {
  %1 = rmo.mo.scatter<() -> S0, S1>(%input, %updates, %indices) [axis = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1], f32>, !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %1 : !mo.tensor<[S0, S1], f32>
}

// -----

mo.graph @scatter_bad_input_dtype<D0, D1, U0, U1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1], f16>, %indices: !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@+1 {{'rmo.mo.scatter' op failed to verify that all of {input, updates, result} have same dtype}}
  %1 = rmo.mo.scatter(%input, %updates, %indices) [axis = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1], f16>, !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @scatter_bad_result_dtype<D0, D1, U0, U1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1], f32>, %indices: !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error@+1 {{'rmo.mo.scatter' op failed to verify that all of {input, updates, result} have same dtype}}
  %1 = rmo.mo.scatter(%input, %updates, %indices) [axis = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1], f32>, !mo.tensor<[I0, I1], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %1 : !mo.tensor<[D0, D1], f16>
}

// -----

mo.graph @scatter_bad_indices_dtype<D0, D1, U0, U1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1], f32>, %indices: !mo.tensor<[I0, I1], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error@+1 {{'rmo.mo.scatter' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<[I0, I1], f32>'}}
  %1 = rmo.mo.scatter(%input, %updates, %indices) [axis = 1] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1], f32>, !mo.tensor<[I0, I1], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
