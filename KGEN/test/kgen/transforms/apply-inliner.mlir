// RUN: kgen-opt -verify-parameters -apply-inliner -split-input-file %s | FileCheck %s

kgen.generator @trivial<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  kgen.return %arg0 : !kgen.paramref<T>
}

// CHECK-LABEL: kgen.generator @trivial_exprs
kgen.generator @trivial_exprs() {
  // CHECK-NEXT: constant = <2>
  kgen.param.constant = <apply(:(index) -> index @trivial<:type index>, 2)>
  kgen.return
}

// -----

kgen.generator @fwd_reg<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  kgen.return %arg0 : !kgen.paramref<T>
}

kgen.generator @fwd_reg_init_self<T: type>(%arg0: !kgen.pointer<T> init_self, %arg1: !kgen.paramref<T>) -> !kgen.none {
  pop.store %arg1, %arg0 : !kgen.pointer<T>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @fwd_reg_byref_result_store_first<T: type>(%arg0: !kgen.paramref<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @fwd_reg_byref_result_store_second<T: type>(%arg0: !kgen.paramref<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  kgen.return %none : !kgen.none
}

kgen.generator @reg_constant<T: type, value: !kgen.paramref<T>>() -> !kgen.paramref<T> {
  %0 = kgen.param.constant: !kgen.paramref<T> = <value>
  kgen.return %0 : !kgen.paramref<T>
}

// CHECK-LABEL: @test_param_inline
kgen.generator @test_param_inline<param>() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(index) -> index @fwd_reg<:type index>, 1)>
  // CHECK-NEXT: <2>
  kgen.param.constant = <apply_result_slot(:(!kgen.pointer<index> init_self, index) -> !kgen.none @fwd_reg_init_self<:type index>, 2)>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_first<:type index>, 3)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_second<:type index>, 4)>
  // CHECK-NEXT: <5>
  kgen.param.constant = <apply(:() -> index @reg_constant<:type index, 5>)>
  // CHECK-NEXT: <param>
  kgen.param.constant = <apply(:() -> index @reg_constant<:type index, param>)>
  kgen.return
}
