// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute --model-inputs=ones %t.mlir --result-output-style=full | FileCheck %s

// CHECK: Running 'elemwise_add_with_param':
// CHECK-NEXT: 'elemwise_add_with_param' returned tensor<8xsi32> [8, 8, 8, 8, 8, 8, 8, 8]
mo.graph @elemwise_add_with_param(%0 : !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32> {
  %1 = mo.custom [symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 7}](%0) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  mo.output %1 : !mo.tensor<[8], si32>
}
