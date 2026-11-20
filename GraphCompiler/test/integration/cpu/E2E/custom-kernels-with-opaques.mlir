// TODO(GEX-2735): Support mo.opaque
// XFAIL: *

// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// DISABLED-CHECK-LABEL: Running 'scalar_to_scalar'
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__init__ 42
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__moveinit__ 42
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__del__ 42
// DISABLED-CHECK{LITERAL}: 'scalar_to_scalar' returned tensor<1xsi32> [42]

// mo.graph @scalar_to_scalar() -> (!mo.tensor<[1], si32>) {
//   %0 = mo.constant <#M.dense_array<42> : tensor<si32>> : !mo.tensor<[1], si32>
//   %1 = mo.custom [symbol = "tensor_to_my_custom_scalar"](%0) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalar">
//   %2 = mo.custom [symbol = "my_custom_scalar_to_tensor"](%1) : (!mo.opaque<"MyCustomScalar">) -> !mo.tensor<[1], si32>
//   mo.output %2 : !mo.tensor<[1], si32>
// }


// DISABLED-CHECK-LABEL: 'tensor_scaled_with_custom_scalar':
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__init__ 2
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__moveinit__ 2
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__del__ 2
// DISABLED-CHECK{LITERAL}: 'tensor_scaled_with_custom_scalar' returned tensor<1x2x2xsi32> [2, 4, 6, 8]

// mo.graph @tensor_scaled_with_custom_scalar() -> (!mo.tensor<[1, 2, 2], si32>) {
//   %0 = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[1], si32>
//   %1 = mo.custom [symbol = "tensor_to_my_custom_scalar"](%0) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalar">
//   %2 = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi32>> : !mo.tensor<[1, 2, 2], si32>
//   %3 = mo.custom [symbol = "scale_with_my_custom_scalar"](%2, %1) : (!mo.tensor<[1, 2, 2], si32>, !mo.opaque<"MyCustomScalar">) -> !mo.tensor<[1, 2, 2], si32>
//   mo.output %3 : !mo.tensor<[1, 2, 2], si32>
// }

// DISABLED-CHECK-LABEL: 'tensor_scaled_with_custom_scalar_ndbuff':
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__init__ 2
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__moveinit__ 2
// DISABLED-CHECK{LITERAL}: MyCustomScalar.__del__ 2
// DISABLED-CHECK{LITERAL}: 'tensor_scaled_with_custom_scalar_ndbuff' returned tensor<1x2x2xsi32> [2, 4, 6, 8]

// mo.graph @tensor_scaled_with_custom_scalar_ndbuff() -> (!mo.tensor<[?, ?, ?], si32>) {
//   %0 = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[1], si32>
//   %1 = mo.custom [symbol = "tensor_to_my_custom_scalar_ndbuff"](%0) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalar">
//   %2 = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi32>> : !mo.tensor<[1, 2, 2], si32>
//   %3 = mo.custom [symbol = "scale_with_my_custom_scalar_ndbuff"](%2, %1) : (!mo.tensor<[1, 2, 2], si32>, !mo.opaque<"MyCustomScalar">) -> !mo.tensor<[?, ?, ?], si32>
//   mo.output %3 : !mo.tensor<[?, ?, ?], si32>
// }

// CHECK-LABEL: Running 'scalar_to_scalar_ndbuff_reg'
// CHECK{LITERAL}: MyCustomScalarReg.__init__ 42
// CHECK{LITERAL}: MyCustomScalarReg.__deinit__ 42
// CHECK{LITERAL}: 'scalar_to_scalar_ndbuff_reg' returned tensor<1xsi32> [42]

mo.graph @scalar_to_scalar_ndbuff_reg() -> (!mo.tensor<[1], si32>) {
  %0 = mo.constant <#M.dense_array<42> : tensor<si32>> : !mo.tensor<[1], si32>
  %1 = mo.custom [symbol = "buff_to_my_custom_scalar_reg", device = #M.device_ref<"cpu", 0>](%0) : (!mo.tensor<[1], si32>) -> !mo.opaque<"MyCustomScalarReg">
  %2 = mo.custom [symbol = "my_custom_scalar_reg_to_buff", device = #M.device_ref<"cpu", 0>](%1) : (!mo.opaque<"MyCustomScalarReg">) -> !mo.tensor<[1], si32>
  mo.output %2 : !mo.tensor<[1], si32>
}
