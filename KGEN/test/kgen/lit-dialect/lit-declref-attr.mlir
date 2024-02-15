// RUN: kgen-opt %s | FileCheck %s
// RUN: kgen-opt -mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: @declref_sugar
// GENERIC-LABEL: kgen.generator
kgen.generator @declref_sugar() {
  // CHECK-NEXT: <{}>
  // GENERIC-NEXT: #lit.struct<{}>
  kgen.param.constant: @S = <{}>
  // CHECK-NEXT: <{1}>
  // GENERIC-NEXT: #lit.struct<{value = 1}>
  kgen.param.constant: @S = <{1}>
  // CHECK-NEXT: <{:dtype f32}>
  // GENERIC-NEXT: #lit.struct<{value: dtype = f32}>
  kgen.param.constant: @S = <{:dtype f32}>
  // GENERIC-NEXT: #lit.struct<{a = 1, b: dtype = f32}>
  // CHECK-NEXT: <{a = 1, b: dtype = f32}>
  kgen.param.constant: @S = <{a = 1, b: dtype = f32}>
  kgen.return
}
