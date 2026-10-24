// RUN: mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" &> %t.txt || true
// RUN: cat %t.txt | FileCheck %s

// Let's only validate that the message from the `constrained` statement in Mojo is displayed,
// since that is the most important part. We could consider also validating source code file
// paths and line numbers, but that can be fragile, as other tests to the test source code might
// cause this test to fail.

// CHECK: note: constraint failed: Expected constraint failure for error message testing
mo.graph @custom_op_failing_constraint() -> !mo.tensor<[1], si32> {
  %data = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %0 = mo.custom [symbol = "mo.test.failing_constraint", device = #M.device_ref<"cpu", 0>](%data) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1], si32>
  mo.output  %0 : !mo.tensor<[1], si32>
}
