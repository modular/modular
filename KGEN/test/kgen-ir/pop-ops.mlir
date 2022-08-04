// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.kernel @pop_constant() -> !meta.scalar<f32> {
kgen.kernel @pop_constant() -> !meta.scalar<f32> {
  // CHECK-NEXT: %cst = pop.constant(32 : si64) : <si64>
  %0 = pop.constant(32 : si64) : !meta.scalar<si64>
  // CHECK-NEXT: %cst_0 = pop.constant(3.200000e+01 : f32) : <f32>
  %1 = pop.constant(32.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: kgen.return  %cst_0 : !meta.scalar<f32>
  kgen.return %1 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_add() -> !meta.scalar<f32> {
kgen.kernel @pop_add() -> !meta.scalar<f32> {
  // CHECK-NEXT: %cst = pop.constant(4.000000e+00 : f32) : <f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %cst_0 = pop.constant(6.000000e+00 : f32) : <f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %0 = pop.add %cst, %cst_0 : <f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  // CHECK-NEXT: kgen.return  %0 : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_sub() -> !meta.scalar<f32> {
kgen.kernel @pop_sub() -> !meta.scalar<f32> {
  // CHECK-NEXT: %cst = pop.constant(4.000000e+00 : f32) : <f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %cst_0 = pop.constant(6.000000e+00 : f32) : <f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %0 = pop.add %cst, %cst_0 : <f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  // CHECK-NEXT: kgen.return  %0 : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_mul() -> !meta.scalar<f32> {
kgen.kernel @pop_mul() -> !meta.scalar<f32> {
  // CHECK-NEXT: %cst = pop.constant(4.000000e+00 : f32) : <f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %cst_0 = pop.constant(6.000000e+00 : f32) : <f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %0 = pop.add %cst, %cst_0 : <f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  // CHECK-NEXT: kgen.return  %0 : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}
