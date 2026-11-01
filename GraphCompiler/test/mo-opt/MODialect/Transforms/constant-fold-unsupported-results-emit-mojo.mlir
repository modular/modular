// TODO(GEX-2735): Support mo.opaque types and parametrization for EmitMojo
// XFAIL: *
// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" %s --mlir-print-ir-after="constant-fold" 2>&1 | FileCheck %s

// COM: This needs to be a separate test since the constant fold pass (which is invoked directly in other tests) does not have a kernel-library-paths option
// COM: Do not constant fold ops with constant operands if they have unsupported result types

// CHECK-LABEL: constant_fold_unsupported_result_types
mo.graph @constant_fold_unsupported_result_types() -> (!mo.opaque<"MyCustomScalarReg">) {
  %0 = mo.constant <#M.dense_array<42> : tensor<si32>> : !mo.tensor<[1], si32>
  %ch0 = mo.chain.create()

  // CHECK: mo.custom [symbol = "buff_to_my_custom_scalar_reg"]
  %1 = mo.custom [symbol = "buff_to_my_custom_scalar_reg", device = #M.device_ref<"cpu", 0>](%0) : (!mo.tensor<[1], si32>) -> (!mo.opaque<"MyCustomScalarReg">)

  mo.output %1 : !mo.opaque<"MyCustomScalarReg">
}
