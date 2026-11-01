// RUN: mo-opt --split-input-file --verify-diagnostics %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @random_normal_ok
mo.graph @random_normal_ok<() -> D0, D1>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {

  // CHECK: mo.random.normal
  %0 = mo.random.normal<() -> S0, S1>(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}

// -----

// CHECK-LABEL: mo.graph @random_uniform_ok
mo.graph @random_uniform_ok<() -> D0, D1>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1], f32> {

  // CHECK: mo.random.uniform
  %0 = mo.random.uniform<() -> S0, S1>(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[S0, S1], f32>
  mo.output<S0, S1> %0 : !mo.tensor<[S0, S1], f32>
}
