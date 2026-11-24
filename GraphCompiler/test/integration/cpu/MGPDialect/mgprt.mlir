// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_pkg"" -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

// CHECK: MGP_RT_TEST
mo.graph @test() -> (!mo.tensor<[1], f32>) {
  %0 = mo.custom [symbol = "mgprt_test_func", device = #M.device_ref<"cpu", 0>]() : () -> !mo.tensor<[1], f32>
  mo.output %0 : !mo.tensor<[1], f32>
}
