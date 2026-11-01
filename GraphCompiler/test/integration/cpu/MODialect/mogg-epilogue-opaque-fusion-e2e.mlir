// TODO(GEX-2314): Support opaque output fusion to enable this test.
// REQUIRES: disabled
// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck --check-prefix=CHECK-MT %s

// CHECK-MT: [1.0, 1.0, 1.0, 1.0, 1.0]
// CHECK-MT: [1.0, 3.0, 3.0, 3.0, 1.0]
// CHECK-MT: [1.0, 3.0, 3.0, 3.0, 1.0]
// CHECK-MT: [1.0, 3.0, 3.0, 3.0, 1.0]
// CHECK-MT: [1.0, 1.0, 1.0, 1.0, 1.0]

!tt1 = !mo.tensor<[5, 5], f32>
!tt2 = !mo.tensor<[3, 3], f32>
!ot = !mo.opaque<"TensorWrapper">
!ct = !mo.chain

mo.graph @matmul_opaque_fusion(%input: !tt1, %matmul_input1: !tt2, %matmul_input2: !tt2) {

  %ch0 = mo.chain.create()

  %result = mo.custom [symbol = "to_tensor_wrapper", device = #M.device_ref<"cpu", 0>] (%input) : (!tt1) -> !ot

  %matmul_result = mo.matmul(%matmul_input1, %matmul_input2) : (!tt2, !tt2) -> !tt2
  %ch1 = mo.custom [symbol = "fused_write_to_tensor_wrapper", device = #M.device_ref<"cpu", 0>] (%result, %matmul_result, %ch0) : (!ot, !tt2, !ct) -> !ct

  %ch2 = mo.custom [symbol = "print_tensor_wrapper", device = #M.device_ref<"cpu", 0>] (%result, %ch1) : (!ot, !ct) -> !ct

  mo.output
}
