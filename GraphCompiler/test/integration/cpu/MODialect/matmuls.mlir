// RUN: mt --execute --result-output-style=full %s | FileCheck %s

mo.graph @identity() -> (!mo.tensor<[3, 3], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0> : tensor<3x3xf32>} : () -> !mo.tensor<[3, 3], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3xf32>} : () -> !mo.tensor<[3, 3], f32>
  %output = mo.matmul(%identity, %value) : (!mo.tensor<[3, 3], f32>, !mo.tensor<[3, 3], f32>) -> !mo.tensor<[3, 3], f32>
  mo.output %output : !mo.tensor<[3, 3], f32>
}
// CHECK{LITERAL}: 'identity' returned tensor<3x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]


mo.graph @identity2() -> (!mo.tensor<[4, 4], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0> : tensor<4x4xf32>} : () -> !mo.tensor<[4, 4], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0> : tensor<4x4xf32>} : () -> !mo.tensor<[4, 4], f32>
  %output = mo.matmul(%identity, %value) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  mo.output %output : !mo.tensor<[4, 4], f32>
}
// CHECK{LITERAL}: 'identity2' returned tensor<4x4xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]


mo.graph @nonSquare() -> (!mo.tensor<[4, 2], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0> : tensor<4x4xf32>} : () -> !mo.tensor<[4, 4], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<4x2xf32>} : () -> !mo.tensor<[4, 2], f32>
  %output = mo.matmul(%identity, %value) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 2], f32>
  mo.output %output : !mo.tensor<[4, 2], f32>
}
// CHECK{LITERAL}: 'nonSquare' returned tensor<4x2xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]


mo.graph @matmulDot() -> (!mo.tensor<[1, 1], f32>) {
  %A = "mo.constant"() {value = #M.dense_array<2.0, 3.0, 4.0, 5.0> : tensor<1x4xf32>} : () -> !mo.tensor<[1, 4], f32>
  %B = "mo.constant"() {value = #M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<4x1xf32>} : () -> !mo.tensor<[4, 1], f32>
  %output = mo.matmul(%A, %B) : (!mo.tensor<[1, 4], f32>, !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], f32>
  mo.output %output : !mo.tensor<[1, 1], f32>
}
// CHECK{LITERAL}: 'matmulDot' returned tensor<1x1xf32> [1.400000e+01]


mo.graph @matmulDot1() -> (!mo.tensor<[1, 1], f32>) {
  %A = "mo.constant"() {value = #M.dense_array<2.0, 3.0, 4.0, 5.0> : tensor<1x4xf32>} : () -> !mo.tensor<[1, 4], f32>
  %B = "mo.constant"() {value = #M.dense_array<0.5, 0.5, 0.5, 0.5> : tensor<4x1xf32>} : () -> !mo.tensor<[4, 1], f32>
  %output = mo.matmul(%A, %B) : (!mo.tensor<[1, 4], f32>, !mo.tensor<[4, 1], f32>) -> !mo.tensor<[1, 1], f32>
  mo.output %output : !mo.tensor<[1, 1], f32>
}
// CHECK{LITERAL}: 'matmulDot1' returned tensor<1x1xf32> [7.000000e+00


mo.graph @vectorOuterProduct() -> (!mo.tensor<[4, 3], f32>) {
  %A = "mo.constant"() {value = #M.dense_array<2.0, 3.0, 4.0, 5.0> : tensor<4x1xf32>} : () -> !mo.tensor<[4, 1], f32>
  %B = "mo.constant"() {value = #M.dense_array<1.0, 0.5, 0.25> : tensor<1x3xf32>} : () -> !mo.tensor<[1, 3], f32>
  %output = mo.matmul(%A, %B) : (!mo.tensor<[4, 1], f32>, !mo.tensor<[1, 3], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %output : !mo.tensor<[4, 3], f32>
}
// CHECK{LITERAL}: 'vectorOuterProduct' returned tensor<4x3xf32> [2.000000e+00, 1.000000e+00, 5.000000e-01, 3.000000e+00, 1.500000e+00, 7.500000e-01, 4.000000e+00, 2.000000e+00, 1.000000e+00, 5.000000e+00, 2.500000e+00, 1.250000e+00]
