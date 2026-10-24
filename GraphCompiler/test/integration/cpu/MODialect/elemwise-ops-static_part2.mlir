// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @exp() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 0.0, 1.0, 2.0, 3.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.exp(%0) : !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'exp' returned tensor<5xf32> [3.678795e-01, 1.000000e+00, 2.718282e+00, 7.389056e+00, 2.008554e+01]


mo.graph @log() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.log(%0) : !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'log' returned tensor<5xf32> [0.000000e+00, 6.931472e-01, 1.098612e+00, 1.386294e+00, 1.609438e+00]


mo.graph @negative() -> (!mo.tensor<[4], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, -2.0, 3.0, -4.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %1 = mo.negative(%0) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
// CHECK{LITERAL}: 'negative' returned tensor<4xf32> [-1.000000e+00, 2.000000e+00, -3.000000e+00, 4.000000e+00]


mo.graph @tanh() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<-10.0, -1.0, -0.5, 0.5, 1.0, 10.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.tanh(%0) : !mo.tensor<[6], f32>
  mo.output %1 : !mo.tensor<[6], f32>
}
// CHECK{LITERAL}: 'tanh' returned tensor<6xf32> [-1.000000e+00, -7.615942e-01, -4.621172e-01, 4.621172e-01, 7.615942e-01, 1.000000e+00]

mo.graph @sqrt() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<0.5, 1.0, 2.2, 3.0, 7.9, 10.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.sqrt(%0) : !mo.tensor<[6], f32>
  mo.output %1 : !mo.tensor<[6], f32>
}
// CHECK: 'sqrt' returned tensor<6xf32> [7.0710{{.*}}e-01, 1.0000{{.*}}e+00, 1.4832{{.*}}e+00, 1.7320{{.*}}e+00, 2.8106{{.*}}e+00, 3.1622{{.*}}e+00]


mo.graph @rsqrt() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<0.5, 1.0, 2.2, 3.0, 7.9, 10.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.rsqrt(%0) : !mo.tensor<[6], f32>
  mo.output %1 : !mo.tensor<[6], f32>
}
// CHECK: 'rsqrt' returned tensor<6xf32> [1.4142{{.*}}e+00, 1.0000{{.*}}e+00, 6.7419{{.*}}e-01, 5.7735{{.*}}e-01, 3.5578{{.*}}e-01, 3.1622{{.*}}e-01]


mo.graph @pow() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, -1.0, 2.0, -2.0, 4.0, -4.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, -1.0, -2.0, -3.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %2 = mo.pow(%0 : !mo.tensor<[6], f32>, %1 : !mo.tensor<[6], f32>) : !mo.tensor<[6], f32>
  mo.output %2 : !mo.tensor<[6], f32>
}
// CHECK: 'pow' returned tensor<6xf32> [1.0000{{.*}}e+00, 1.0000{{.*}}e+00, 8.0000{{.*}}e+00, -5.0000{{.*}}e-01, 6.2500{{.*}}e-02, -1.5625{{.*}}e-02]


mo.graph @pow_int_exponent() -> (!mo.tensor<[6], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, -1.0, 2.0, -2.0, 4.0, -4.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %2 = mo.pow(%0 : !mo.tensor<[6], f32>, %1 : !mo.tensor<[6], si32>) : !mo.tensor<[6], f32>
  mo.output %2 : !mo.tensor<[6], f32>
}
// CHECK: 'pow_int_exponent' returned tensor<6xf32> [1.0000{{.*}}e+00, 1.0000{{.*}}e+00, 8.0000{{.*}}e+00, 1.6000{{.*}}e+01, 1.0240{{.*}}e+03, 4.0960{{.*}}e+03]


mo.graph @sin() -> (!mo.tensor<[4], f32>) {
  %0 = mo.constant <#M.dense_array<0.0, 1.570796, 1.0, -1.0>: tensor<4xf32>> : !mo.tensor<[4], f32>
  %res = mo.sin(%0) : !mo.tensor<[4], f32>
  mo.output %res : !mo.tensor<[4], f32>
}
// CHECK: 'sin' returned tensor<4xf32> [0.0000{{.*}}e+00, 1.0000{{.*}}e+00, 8.4147{{.*}}e-01, -8.4147{{.*}}e-01]

mo.graph @cos() -> (!mo.tensor<[4], f32>) {
  %0 = mo.constant <#M.dense_array<0.0, 1.570796, 1.0, -1.0>: tensor<4xf32>> : !mo.tensor<[4], f32>
  %res = mo.cos(%0) : !mo.tensor<[4], f32>
  mo.output %res : !mo.tensor<[4], f32>
}
// CHECK: 'cos' returned tensor<4xf32> [1.0000{{.*}}e+00, 3.1391{{.*}}e-07, 5.4030{{.*}}e-01, 5.4030{{.*}}e-01]

mo.graph @cast() -> (!mo.tensor<[6], si32>) {
  %0 = mo.constant <#M.dense_array<-10.0, -1.0, -0.3, 0.3, 1.0, 10.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %1 = mo.cast(%0) : (!mo.tensor<[6], f32>) -> !mo.tensor<[6], si32>
  mo.output %1 : !mo.tensor<[6], si32>
}
// CHECK: 'cast' returned tensor<6xsi32> [-10, -1, 0, 0, 1, 10]

mo.graph @ceil() -> (!mo.tensor<[8], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-3.5, -3.1, -2.5, -2.1, 2.1, 2.5, 3.1, 3.5> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>
  %1 = mo.ceil(%0) : !mo.tensor<[8], f32>
  mo.output %1 : !mo.tensor<[8], f32>
}
// CHECK: 'ceil' returned tensor<8xf32> [-3.000000e+00, -3.000000e+00, -2.000000e+00, -2.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]

mo.graph @floor() -> (!mo.tensor<[8], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-3.5, -3.1, -2.5, -2.1, 2.1, 2.5, 3.1, 3.5> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>
  %1 = mo.floor(%0) : !mo.tensor<[8], f32>
  mo.output %1 : !mo.tensor<[8], f32>
}
// CHECK: 'floor' returned tensor<8xf32> [-4.000000e+00, -4.000000e+00, -3.000000e+00, -3.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00]

mo.graph @trunc() -> (!mo.tensor<[8], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-3.5, -3.1, -2.5, -2.1, 2.1, 2.5, 3.1, 3.5> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>
  %1 = mo.trunc(%0) : !mo.tensor<[8], f32>
  mo.output %1 : !mo.tensor<[8], f32>
}
// CHECK: 'trunc' returned tensor<8xf32> [-3.000000e+00, -3.000000e+00, -2.000000e+00, -2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00]

mo.graph @round() -> (!mo.tensor<[8], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-3.5, -3.1, -2.5, -2.1, 2.1, 2.5, 3.1, 3.5> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>
  %1 = mo.round(%0) : !mo.tensor<[8], f32>
  mo.output %1 : !mo.tensor<[8], f32>
}
// CHECK: 'round' returned tensor<8xf32> [-4.000000e+00, -3.000000e+00, -2.000000e+00, -2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]
