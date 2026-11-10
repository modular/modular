// RUN: mo-opt %s -o %t.mlir --gc-pipeline="default-device-label=cpu constant-fold=false kernel-library-paths=%kernel_api_test_pkg" 2>&1
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// Test printing of a trait argument's TileTensor projection
// CHECK{LITERAL}: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
mo.graph @print() -> !mo.tensor<[2, 4], f32> {
  %range = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>> : !mo.tensor<[2, 4], f32>
  %out = mo.custom [symbol = "print_tensor", device = #M.device_ref<"cpu", 0>] (%range) : (!mo.tensor<[2, 4], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %out : !mo.tensor<[2, 4], f32>
}
