// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

mo.graph @si32_mul() -> (!mo.tensor<[6], si32>) {
  %0 = mo.constant <#M.dense_array<1, -2, 14, 4, -2, -6> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %1 = mo.constant <#M.dense_array<-4, 563, 129, 6, -4, 1> : tensor<6xsi32>> :!mo.tensor<[6], si32>
  %2 = mo.mul(%0, %1) : !mo.tensor<[6], si32>
  mo.output %2 : !mo.tensor<[6], si32>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'si32_mul':
// CHECK{LITERAL}: 'si32_mul' returned tensor<6xsi32> [-4, -1126, 1806, 24, 8, -6]

mo.graph @si32_div(%arg0 : !mo.tensor<[6], si32>) -> (!mo.tensor<[6], si32>) {
  %0 = mo.add(%arg0, %arg0) : !mo.tensor<[6], si32>
  %1 = mo.add(%0, %0) : !mo.tensor<[6], si32>

  %cst = "mo.constant"() {value = #M.dense_array<16, 4563, 128, -8, 3, 4> : tensor<6xsi32>} : () -> !mo.tensor<[6], si32>
  %2 = mo.div(%cst, %1) : !mo.tensor<[6], si32>
  mo.output %2 : !mo.tensor<[6], si32>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'si32_div':
// CHECK{LITERAL}: 'si32_div' returned tensor<6xsi32> [4, 1140, 32, -2, 0, 1]


mo.graph @fused_si32() -> (!mo.tensor<[6], si32>) {
  %0 = "mo.constant"() {value = #M.dense_array<1, -2, 14, 4, -2, -6> : tensor<6xsi32>} : () -> !mo.tensor<[6], si32>
  %1 = "mo.constant"() {value = #M.dense_array<-4, 563, 129, 6, -4, 1> : tensor<6xsi32>} : () -> !mo.tensor<[6], si32>
  %2 = mo.mul(%0, %1) : !mo.tensor<[6], si32>
  %3 = mo.relu(%2) : !mo.tensor<[6], si32>
  mo.output %3 : !mo.tensor<[6], si32>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'fused_si32':
// CHECK{LITERAL}: 'fused_si32' returned tensor<6xsi32> [0, 0, 1806, 24, 8, 0]

mo.graph @fused_si16() -> (!mo.tensor<[6], si16>) {
  %0 = "mo.constant"() {value = #M.dense_array<1, -2, 14, 4, -2, -6> : tensor<6xsi16>} : () -> !mo.tensor<[6], si16>
  %1 = "mo.constant"() {value = #M.dense_array<-4, 563, 129, 6, -4, 1> : tensor<6xsi16>} : () -> !mo.tensor<[6], si16>
  %2 = mo.mul(%0, %1) : !mo.tensor<[6], si16>
  %3 = mo.relu(%2) : !mo.tensor<[6], si16>
  mo.output %3 : !mo.tensor<[6], si16>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'fused_si16':
// CHECK{LITERAL}: 'fused_si16' returned tensor<6xsi16> [0, 0, 1806, 24, 8, 0]

mo.graph @fused_ui16() -> (!mo.tensor<[6], ui16>) {
  %0 = "mo.constant"() {value = #M.dense_array<1, 0, 14, 4, 2, 0> : tensor<6xui16>} : () -> !mo.tensor<[6], ui16>
  %1 = "mo.constant"() {value = #M.dense_array<0, 563, 129, 6, 4, 1> : tensor<6xui16>} : () -> !mo.tensor<[6], ui16>
  %2 = mo.mul(%0, %1) : !mo.tensor<[6], ui16>
  %3 = mo.relu(%2) : !mo.tensor<[6], ui16>
  mo.output %3 : !mo.tensor<[6], ui16>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'fused_ui16':
// CHECK{LITERAL}: 'fused_ui16' returned tensor<6xui16> [0, 0, 1806, 24, 8, 0]

mo.graph @many_bcasts_si32() -> !mo.tensor<[2, 2, 3], si32> {
  %0 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12> : tensor<12xsi32>> : !mo.tensor<[2, 2, 3], si32>
  %1 = mo.constant <#M.dense_array<3, 0> : tensor<2xsi32>> : !mo.tensor<[2, 1, 1], si32>
  %2 = mo.constant <#M.dense_array<5, 25> : tensor<2xsi32>> : !mo.tensor<[2, 1], si32>
  %3 = mo.constant <#M.dense_array<1, 7, 8> : tensor<3xsi32>> : !mo.tensor<[3], si32>

  %bcast = mo.static.broadcast_to(%1) : !mo.tensor<[2, 1, 1], si32> -> !mo.tensor<[2, 2, 3], si32>
  %bcast2 = mo.static.broadcast_to(%2) : !mo.tensor<[2, 1], si32> -> !mo.tensor<[2, 2, 3], si32>
  %bcast3 = mo.static.broadcast_to(%3) : !mo.tensor<[3], si32> -> !mo.tensor<[2, 2, 3], si32>

  %out = mo.add(%bcast, %0) : !mo.tensor<[2, 2, 3], si32>
  %out2 = mo.mul(%bcast2, %out) : !mo.tensor<[2, 2, 3], si32>
  %out3 = mo.sub(%out2, %bcast3) : !mo.tensor<[2, 2, 3], si32>

  mo.output %out3 : !mo.tensor<[2, 2, 3], si32>
}

// To produce reference values:
// x = np.arange(1, 13).reshape(2, 2, 3)
// y = np.asarray([3, 0]).reshape(2, 1, 1)
// z = np.asarray([5, 25]).reshape(2, 1)
// w = np.asarray([1, 7, 8]).reshape(1, 1, 3)
// print(((y + x) * z) - w)

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'many_bcasts_si32':
// CHECK{LITERAL}: 'many_bcasts_si32' returned tensor<2x2x3xsi32> [19, 18, 22, 174, 193, 217, 34, 33, 37, 249, 268, 292]

mo.graph @fused_si64() -> (!mo.tensor<[6], si64>) {
  %0 = "mo.constant"() {value = #M.dense_array<1, -2, 14, 4, -2, -6> : tensor<6xsi64>} : () -> !mo.tensor<[6], si64>
  %1 = "mo.constant"() {value = #M.dense_array<-4, 563, 129, 6, -4, 1> : tensor<6xsi64>} : () -> !mo.tensor<[6], si64>
  %2 = mo.mul(%0, %1) : !mo.tensor<[6], si64>
  %3 = mo.relu(%2) : !mo.tensor<[6], si64>
  mo.output %3 : !mo.tensor<[6], si64>
}

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'fused_si64':
// CHECK{LITERAL}: 'fused_si64' returned tensor<6xsi64> [0, 0, 1806, 24, 8, 0]

mo.graph @many_bcasts_si64() -> !mo.tensor<[2, 2, 3], si64> {
  %0 = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12> : tensor<12xsi64>> : !mo.tensor<[2, 2, 3], si64>
  %1 = mo.constant <#M.dense_array<3, 0> : tensor<2xsi64>> : !mo.tensor<[2, 1, 1], si64>
  %2 = mo.constant <#M.dense_array<5, 25> : tensor<2xsi64>> : !mo.tensor<[2, 1], si64>
  %3 = mo.constant <#M.dense_array<1, 7, 8> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  %bcast = mo.static.broadcast_to(%1) : !mo.tensor<[2, 1, 1], si64> -> !mo.tensor<[2, 2, 3], si64>
  %bcast2 = mo.static.broadcast_to(%2) : !mo.tensor<[2, 1], si64> -> !mo.tensor<[2, 2, 3], si64>
  %bcast3 = mo.static.broadcast_to(%3) : !mo.tensor<[3], si64> -> !mo.tensor<[2, 2, 3], si64>

  %out = mo.add(%bcast, %0) : !mo.tensor<[2, 2, 3], si64>
  %out2 = mo.mul(%bcast2, %out) : !mo.tensor<[2, 2, 3], si64>
  %out3 = mo.sub(%out2, %bcast3) : !mo.tensor<[2, 2, 3], si64>

  mo.output %out3 : !mo.tensor<[2, 2, 3], si64>
}

// To produce reference values:
// x = np.arange(1, 13).reshape(2, 2, 3)
// y = np.asarray([3, 0]).reshape(2, 1, 1)
// z = np.asarray([5, 25]).reshape(2, 1)
// w = np.asarray([1, 7, 8]).reshape(1, 1, 3)
// print(((y + x) * z) - w)

// CHECK-IR: mgp.generic.execute
// CHECK-LABEL:--- Running 'many_bcasts_si64':
// CHECK{LITERAL}: 'many_bcasts_si64' returned tensor<2x2x3xsi64> [19, 18, 22, 174, 193, 217, 34, 33, 37, 249, 268, 292]


mo.graph @fused_cast(%arg0: !mo.tensor<[2, 2, 2], si32>, %arg1: !mo.tensor<[2, 2, 2], si32>) -> (!mo.tensor<[2, 2, 2], f32>) {
  %0 = mo.cast(%arg0) : (!mo.tensor<[2, 2, 2], si32>) -> !mo.tensor<[2, 2, 2], f32>
  %1 = mo.cast(%arg1) : (!mo.tensor<[2, 2, 2], si32>) -> !mo.tensor<[2, 2, 2], f32>
  %cst = mo.constant <#M.dense_array<2.500000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %cst_bcast = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2, 2], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[2, 2, 2], f32>
  %3 = mo.mul(%2, %cst_bcast) : !mo.tensor<[2, 2, 2], f32>
  mo.output %3 : !mo.tensor<[2, 2, 2], f32>
}

// CHECK-IR: mgp.generic.execute
// CHECK{LITERAL}: 'fused_cast' returned tensor<2x2x2xf32> [5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01]
