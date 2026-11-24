// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s


// CHECK: mo.graph @exp(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> isSubgraph {
mo.graph @exp(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) isSubgraph {
  // CHECK-NEXT: mo.exp(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.exp(%arg0) : !mo.tensor<[4, 5], f32>
  // CHECK-NEXT: %1 = mo.constant.external [name = "foo", align = 16, device = #M.device_ref<"cpu", 0>, isPlaceholder = true] : !mo.tensor<[4, 5], f32>
  %1 = mo.constant.external [name = "foo", align = 16, device = #M.device_ref<"cpu", 0>, isPlaceholder = true] : !mo.tensor<[4, 5], f32>
  // CHECK-NEXT: %2 = mo.add(%0, %1) : !mo.tensor<[4, 5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[4, 5], f32>
  // CHECK-NEXT: mo.output %2 : !mo.tensor<[4, 5], f32>
  mo.output %2 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @main(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @main(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: %0 = mo.call @exp(%arg0) [prefix = "prefix"] : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  %0 = mo.call @exp(%arg0) [prefix = "prefix"] : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}
