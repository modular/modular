// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @variant_create_get
kgen.func @variant_create_get(%a: i32) -> i32 {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  %1 = pop.variant.get %0 : !pop.variant<i32, f32> as i32
  // CHECK: return %arg0
  kgen.return %1 : i32
}
