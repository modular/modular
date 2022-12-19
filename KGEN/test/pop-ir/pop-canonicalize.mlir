// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @variant_create_get
kgen.func @variant_create_get(%a: i32) -> i32 {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  %1 = pop.variant.get %0 : !pop.variant<i32, f32> as i32
  // CHECK: return %arg0
  kgen.return %1 : i32
}


// CHECK-LABEL: @call_indirect_partial_apply
kgen.generator @call_indirect_partial_apply(%fn: (index, i32) -> index, %arg0: index, %arg1: i32) -> index {
  // CHECK-NEXT: %0 = pop.call_indirect %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = pop.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = pop.call_indirect %0(%arg0) : !pop.closure<(index) -> index>
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @partial_apply_of_partial_apply
kgen.generator @partial_apply_of_partial_apply(%fn: (index, i32) -> index, %arg0: index, %arg1: i32) -> !pop.closure<() -> index> {
  // CHECK-NEXT: %0 = pop.partial_apply %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = pop.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = pop.partial_apply %0(%arg0) : !pop.closure<(index) -> index>
  // CHECK-NEXT: return %0
  kgen.return %1 : !pop.closure<() -> index>
}
