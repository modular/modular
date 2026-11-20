// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --model-inputs=ones --result-output-style=full | FileCheck %s

// CHECK-LABEL: Running 'elemwise_add_with_index':
// CHECK: 'elemwise_add_with_index' returned tensor<8xsi32> [1, 2, 3, 4, 5, 6, 7, 8]
mo.graph @elemwise_add_with_index(%0 : !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32> {
   %1 = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%0) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
   mo.output %1 : !mo.tensor<[8], si32>
}
