// RUN: mo-opt --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" %s -o %t_mojo.mlir
// RUN: mt --execute %t_mojo.mlir &> %t_mojo.txt || true
// RUN: cat %t_mojo.txt | FileCheck %s

// CHECK: This is an error
mo.graph @return_error() -> !mo.tensor<[1], si32> {
  %data = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %0 = mo.custom [symbol = "mo.test.return_error", device = #M.device_ref<"cpu", 0>](%data) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1], si32>
  mo.output %0 : !mo.tensor<[1], si32>
}
