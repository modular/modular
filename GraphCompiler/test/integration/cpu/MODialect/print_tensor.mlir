// RUN: mo-opt %s -o %t.mlir --gc-pipeline="default-device-label=cpu constant-fold=false kernel-library-paths=%kernel_api_test_pkg" 2>&1
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// Test printing of ManagedTensorSlice
// CHECK{LITERAL}: ManagedTensorSlice([[1.0, 2.0, 3.0, 4.0],
// CHECK-NEXT{LITERAL}: [5.0, 6.0, 7.0, 8.0]], dtype=float32, shape=2x4)
// CHECK-SAME{LITERAL}: {static_shape = (2, 4), static_strides = (4, 1), dynamic_shape = (2, 4), dynamic_strides = (4, 1),

// Note: alignment is platform specific
// CHECK-SAME: alignment = [[ALIGNMENT:[0-9]+]], address_space = AddressSpace.GENERIC}
mo.graph @print() -> !mo.tensor<[2, 4], f32> {
  %range = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>> : !mo.tensor<[2, 4], f32>
  %out = mo.custom [symbol = "print_tensor", device = #M.device_ref<"cpu", 0>] (%range) : (!mo.tensor<[2, 4], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %out : !mo.tensor<[2, 4], f32>
}
