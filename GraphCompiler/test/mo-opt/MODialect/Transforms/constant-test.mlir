// RUN: mo-opt %s --mo-constant-fold | FileCheck %s


// CHECK: mo.graph @gather_param_no_fold<[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
mo.graph @gather_param_no_fold<D0, D1 -> D2, D3>(
    // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], si32>) -> !mo.tensor<[[[D2]], [[D3]]], si32> {
    %input: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D2, D3], si32> {
  // CHECK-DAG: %[[INDICES:.*]] = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %indices = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[INDICES]]) [axis = 0] : (!mo.tensor<[[[D0]], [[D1]]], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si32>
  %1 = mo.gather<() -> D4, D5>(%input, %indices) [axis = 0] : (!mo.tensor<[D0, D1], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[D4, D5], si32>
  // CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], si32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], si32>
}
