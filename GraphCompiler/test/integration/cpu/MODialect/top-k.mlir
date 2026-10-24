// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK: tensor<5x3xsi64> [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22]
// CHECK-NEXT: , tensor<5x3xsi64> [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
mo.graph @bottom_k() -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %sorted = "mo.constant"() {value = #M.dense_array<1> : tensor<1xi1>} : () -> !mo.tensor<[], bool>
  %values, %indices = mo.bottom_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
  mo.output %values, %indices : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
}

// CHECK: tensor<5x3xsi64> [4, 3, 2, 9, 8, 7, 14, 13, 12, 19, 18, 17, 24, 23, 22]
// CHECK: tensor<5x3xsi64> [4, 3, 2, 4, 3, 2, 4, 3, 2, 4, 3, 2, 4, 3, 2]
mo.graph @top_k() -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %sorted = "mo.constant"() {value = #M.dense_array<1> : tensor<1xi1>} : () -> !mo.tensor<[], bool>
  %values, %indices = mo.top_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
  mo.output %values, %indices : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
}

// CHECK: 'top_k_fallback_shape_func' returned tensor<32x3x20x3xsi32>
mo.graph @top_k_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], si64>,
  %sorted : !mo.tensor<[], bool>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>

  mo.invoke_shape_func["mo.top_k"]<() -> D4, D5, D6, D7>(%input, %K, %axis, %sorted)
    [dataDeptTensors = [1 : i32, 2 : i32]] :
    (!mo.tensor<[32, 10, 20, 3], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
