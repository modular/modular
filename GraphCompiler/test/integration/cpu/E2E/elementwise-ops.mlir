// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'add':
// CHECK: 'add' returned tensor<2x3xsi32> [12, 14, 16, 18, 20, 22]
mo.graph @add() -> !mo.tensor<[2, 3], si32> {
  %0 = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %1 = "mo.constant"() {value = #M.dense_array<11, 12, 13, 14, 15, 16> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %2 = mo.add(%0, %1) : !mo.tensor<[2, 3], si32>
  mo.output %2 : !mo.tensor<[2, 3], si32>
}

// CHECK-LABEL: Running 'few_elemwise_ops':
// CHECK: 'few_elemwise_ops' returned tensor<2x3xsi32> [5, 19, 13, -5, 29, 16]
mo.graph @few_elemwise_ops() -> !mo.tensor<[2, 3], si32> {
  %0 = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %1 = "mo.constant"() {value = #M.dense_array<11, 12, 13, 14, 15, 16> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %2 = "mo.constant"() {value = #M.dense_array<1, -1, 1, -1, -1, 1> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %3 = "mo.constant"() {value = #M.dense_array<8, -5, 4, 23, -9, 7> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %4 = mo.add(%0, %1) : !mo.tensor<[2, 3], si32>
  %5 = mo.relu(%2) : !mo.tensor<[2, 3], si32>
  %6 = mo.sub(%4, %3) : !mo.tensor<[2, 3], si32>
  %7 = mo.add(%6, %5) : !mo.tensor<[2, 3], si32>
  mo.output %7 : !mo.tensor<[2, 3], si32>
}

// CHECK-LABEL: Running 'few_elemwise_ops_cast':
// CHECK: 'few_elemwise_ops_cast' returned tensor<2x3xf32> [5.000000e+00, 1.900000e+01, 1.300000e+01, -5.000000e+00, 2.900000e+01, 1.600000e+01]
mo.graph @few_elemwise_ops_cast() -> !mo.tensor<[2, 3], f32> {
  %0 = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %1 = "mo.constant"() {value = #M.dense_array<11, 12, 13, 14, 15, 16> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %2 = "mo.constant"() {value = #M.dense_array<1, -1, 1, -1, -1, 1> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %3 = "mo.constant"() {value = #M.dense_array<8, -5, 4, 23, -9, 7> : tensor<6xsi32>} : () -> !mo.tensor<[2, 3], si32>
  %4 = mo.add(%0, %1) : !mo.tensor<[2, 3], si32>
  %5 = mo.relu(%2) : !mo.tensor<[2, 3], si32>
  %6 = mo.sub(%4, %3) : !mo.tensor<[2, 3], si32>
  %7 = mo.add(%6, %5) : !mo.tensor<[2, 3], si32>
  %8 = mo.cast(%7) : (!mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], f32>
  mo.output %8 : !mo.tensor<[2, 3], f32>
}
