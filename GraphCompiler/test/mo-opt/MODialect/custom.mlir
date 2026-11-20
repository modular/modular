// RUN: mo-opt --canonicalize %s| FileCheck %s

// CHECK-LABEL: mo.graph @custom_dynamic
// CHECK-SAME:  (%[[ARG0:.*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
// CHECK-NEXT:  %[[CUSTOM:.*]] = mo.custom [symbol = "test_custom_op"](%[[ARG0]])
// CHECK-NEXT:  mo.output %[[CUSTOM]]

mo.graph @custom_dynamic(%arg0 : !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mo.custom [symbol = "test_custom_op", device = #M.device_ref<"cpu", 0>](%arg0) : (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}

// CHECK-LABEL: mo.graph @inplace_custom_op
// CHECK-SAME: [[arg0:%.*]]: !mo.buffer<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mo.custom [symbol = "inplace"]([[arg0]], [[ch0]])
mo.graph @inplace_custom_op(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "inplace", device = #M.device_ref<"cpu", 0>] (%arg0, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output %ch1 : !mo.chain
}

// CHECK-LABEL: mo.graph @custom_models_memory_effects
// Custom should not be deleted since it has memory effects
// CHECK: mo.custom
mo.graph @custom_models_memory_effects<()->D>(%arg0: !mo.tensor<[10, 10], f32>, %data: !mo.tensor<[], f32>) -> !mo.tensor<[D], f32> {
  // This result is not used, but op declares parameter so should not be deleted
  %t = mo.custom <() -> A> [symbol = "example_op", device = #M.device_ref<"cpu", 0>] (%arg0) : (!mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32>

  %res = mo.static.broadcast_to(%data): !mo.tensor<[], f32> -> !mo.tensor<[A], f32>
  mo.output<A> %res : !mo.tensor<[A], f32>
}
