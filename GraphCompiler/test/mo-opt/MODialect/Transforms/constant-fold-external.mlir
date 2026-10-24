// RUN: mo-opt --mo-constant-fold %s | FileCheck %s

// Subgraph with constant externals should not be folded or fail
// CHECK-LABEL: @constant_fold_external_with_constant
// CHECK-SAME: ([[ARG0:%.*]]: !mo.tensor<[256, 1], si32>) -> !mo.tensor<[256, 1], si32>
// CHECK: [[CONST0:%.*]] = mo.constant
// CHECK: [[CONST1:%.*]] = mo.constant.external
// CHECK: [[BROADCAST:%.*]] = mo.static.broadcast_to([[CONST0]])
// CHECK: [[MUL:%.*]] = mo.mul([[CONST1]], [[BROADCAST]])
// CHECK: [[ADD:%.*]] = mo.add([[ARG0]], [[MUL]])
// CHECK: mo.output [[ADD]]
mo.graph @constant_fold_external_with_constant(%arg0: !mo.tensor<[256, 1], si32>) -> !mo.tensor<[256, 1], si32> {
  %0 = mo.constant <#M.dense_array<2> : !M.array<1xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si32>
  %1 = mo.constant.external [name = "blob", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[256, 1], si32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si32> -> !mo.tensor<[256, 1], si32>
  %3 = mo.mul(%1, %2) : !mo.tensor<[256, 1], si32>
  %4 = mo.add(%arg0, %3) : !mo.tensor<[256, 1], si32>
  mo.output %4 : !mo.tensor<[256, 1], si32>
}
