// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scatter_nd_ok<D0, D1, U0, U1, U2, I0, I1, I2 -> R0, R1>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[R0, R1], f32> {
  %0 = rmo.mo.scatter_nd<() -> S0, S1>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}

// -----

mo.graph @scatter_nd_static_ok(%arg0: !mo.tensor<[4, 2], f32>, %updates: !mo.tensor<[1, 3, 2], f32>, %indices: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %0 : !mo.tensor<[4, 2], f32>
}

// -----

mo.graph @scatter_nd_bad_updates<D0, D1, U0, U1, U2, I0, I1, I2>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f16>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f16>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @scatter_nd_bad_indices<D0, D1, U0, U1, U2, I0, I1, I2>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<[I0, I1, I2], f32>'}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @scatter_nd_bad_result<D0, D1, U0, U1, U2, I0, I1, I2>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[D0, D1], f16> {
  // expected-error @+1{{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}

// -----

mo.graph @scatter_nd_add_ok<D0, D1, U0, U1, U2, I0, I1, I2 -> R0, R1>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[R0, R1], f32> {
  %0 = rmo.mo.scatter_nd.add<() -> S0, S1>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}

// -----

mo.graph @scatter_nd_max_ok<D0, D1, U0, U1, U2, I0, I1, I2 -> R0, R1>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[R0, R1], f32> {
  %0 = rmo.mo.scatter_nd.max<() -> S0, S1>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}

// -----

mo.graph @scatter_nd_min_ok<D0, D1, U0, U1, U2, I0, I1, I2 -> R0, R1>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[R0, R1], f32> {
  %0 = rmo.mo.scatter_nd.min<() -> S0, S1>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}

// -----

mo.graph @scatter_nd_mul_ok<D0, D1, U0, U1, U2, I0, I1, I2 -> R0, R1>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[U0, U1, U2], f32>, %indices: !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[R0, R1], f32> {
  %0 = rmo.mo.scatter_nd.mul<() -> S0, S1>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[U0, U1, U2], f32>, !mo.tensor<[I0, I1, I2], si64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}
