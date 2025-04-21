// RUN: kgen-opt %s -split-input-file -elaborate-generators -allow-unregistered-dialect | FileCheck %s

kgen.generator @select_pred<*"cmp`2x": i1>() -> !kgen.deferred {
  kgen.param.if <*"cmp`2x"> {
    %0 = kgen.param.constant: !kgen.deferred = <#kgen<deferred #index<cmp_predicate sle>>>
    kgen.return %0 : !kgen.deferred
  } else {
    %0 = kgen.param.constant: !kgen.deferred = <#kgen<deferred #index<cmp_predicate sgt>>>
    kgen.return %0 : !kgen.deferred
  } {elseIsolated, thenIsolated}
  kgen.unreachable
}

// CHECK-LABEL: @"test_select_pred,cmp=0"
// CHECK: %[[CMP_RESULT:.*]] = index.cmp sgt(%arg0, %arg1)
// CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>

// CHECK-LABEL: @"test_select_pred,cmp=1"
// CHECK: %[[CMP_RESULT:.*]] = index.cmp sle(%arg0, %arg1)
// CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
kgen.generator @test_select_pred<cmp: i1>(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  kgen.param.declare select_pred: <i1>() -> !kgen.deferred = <@select_pred>
  kgen.param.apply *"(lifted)apply_0" = [() -> !kgen.deferred: bind_params(:<i1>() -> !kgen.deferred select_pred, cmp)]()
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred} : i1
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

// CHECK-LABEL: @test_elaborate_deferred_op
kgen.generator @test_elaborate_deferred_op(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: %[[CMP_RESULT:.*]] = index.cmp sle(%arg0, %arg1)
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred} : i1
  // CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

kgen.generator export @test(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws {
  %0 = kgen.call @test_select_pred<:i1 1>(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %1 = kgen.call @test_select_pred<:i1 0>(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %2 = kgen.call @test_elaborate_deferred_op(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1

  kgen.return
}
