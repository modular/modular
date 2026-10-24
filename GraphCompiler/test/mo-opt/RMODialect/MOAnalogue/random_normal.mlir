// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @random_normal_ok<() -> D0, D1>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {
  %0 = rmo.mo.random.normal<() -> R0, R1>(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[R0, R1], f32>
  mo.output<R0, R1> %0 : !mo.tensor<[R0, R1], f32>
}

// -----

mo.graph @random_normal_bad_mean<D0, D1>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], si32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #1 must be rank 0 or unranked with float elements, but got '!mo.tensor<[], si32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @random_normal_bad_variance<D0, D1>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f16>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op failed to verify that all of {mean, variance} have same dtype}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f16>, !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @random_normal_bad_size<D0, D1, D2, D3>(%size: !mo.tensor<[D2, D3], f32>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #0 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[D2, D3], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @random_normal_bad_seed<D0, D1, D2, D3>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #3 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[D2, D3], f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}
