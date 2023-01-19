// RUN: kgen-opt %s -allow-unregistered-dialect -lower-kgen-to-llvm=index-bitwidth=64 -lower-scf-to-llvm=index-bitwidth=64 -canonicalize | FileCheck %s
// RUN: kgen-opt %s -allow-unregistered-dialect -lower-kgen-to-llvm=index-bitwidth=64 -lower-scf-to-llvm=index-bitwidth=64 | FileCheck %s --check-prefix=SWITCH

llvm.func @get(i32) -> i32

// CHECK-LABEL: @loop
// CHECK-SAME: %[[INIT:.*]]: i32, %[[LB:.*]]: i64, %[[UB:.*]]: i64, %[[STEP:.*]]: i64
// CHECK:   llvm.br ^bb1(%[[LB]], %[[INIT]] :
// CHECK: ^bb1(%[[I:.*]]: i64, %[[V:.*]]: i32
// CHECK:   %[[COND:.*]] = llvm.icmp "slt" %[[I]], %[[UB]]
// CHECK:   llvm.cond_br %[[COND]], ^bb2, ^bb3
// CHECK: ^bb2:
// CHECK:   %[[V0:.*]] = llvm.call @get(%[[V]])
// CHECK:   %[[NEXT:.*]] = llvm.add %[[I]], %[[STEP]]
// CHECK:   llvm.br ^bb1(%[[NEXT]], %[[V0]]
// CHECK: ^bb3:
// CHECK    llvm.return %[[V]]
kgen.func @loop(%init: i32, %lb: index, %ub: index, %step: index) -> i32 {
  %result = scf.for %i = %lb to %ub step %step iter_args(%v = %init) -> i32 {
    %0 = llvm.call @get(%v) : (i32) -> i32
    scf.yield %0 : i32
  }
  kgen.return %result : i32
}

// CHECK-LABEL: @cond
// CHECK-SAME: %[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK:   llvm.cond_br %[[COND]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[V0:.*]] = llvm.call @get(%[[A]])
// CHECK:   llvm.br ^bb3(%[[V0]]
// CHECK: ^bb2:
// CHECK:   llvm.br ^bb3(%[[B]]
// CHECK: ^bb3(%[[RES:.*]]: i32
// CHECK:   llvm.br ^bb4
// CHECK: ^bb4:
// CHECK:   llvm.return %[[RES]]
kgen.func @cond(%cond: i1, %a: i32, %b: i32) -> i32 {
  %result = scf.if %cond -> i32 {
    %0 = llvm.call @get(%a) : (i32) -> i32
    scf.yield %0 : i32
  } else {
    scf.yield %b : i32
  }
  kgen.return %result : i32
}

// CHECK-LABEL: @while
// CHECK-SAME: %[[INIT:.*]]:
kgen.func @while(%init: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  // CHECK: llvm.br ^bb1(%[[INIT]]
  %result = scf.while (%v = %init) : (!pop.simd<1, f32>) -> !pop.simd<1, f32> {
    // CHECK: ^bb1(%[[V:.*]]: f32
    // CHECK: llvm.cond_br %{{.*}}, ^bb2(%[[V]] : f32), ^bb3
    %condition = "cond"(%v) : (!pop.simd<1, f32>) -> i1
    scf.condition(%condition) %v : !pop.simd<1, f32>
  } do {
  // CHECK: ^bb2(%[[U:.*]]: f32
  ^bb0(%u: !pop.simd<1, f32>):
    // CHECK: llvm.br ^bb1(
    %next = "next"(%u) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
    scf.yield %next : !pop.simd<1, f32>
  }
  // CHECK: ^bb3:
  // CHECK: return %[[V]]
  kgen.return %result : !pop.simd<1, f32>
}

// SWITCH-LABEL: @scf_index_switch
kgen.func @scf_index_switch(%i: index, %a: i32, %b: i32, %c: i32) -> i32 {
  // SWITCH: llvm.switch %1 : i64, ^bb3 [
  // SWITCH-NEXT: 0: ^bb1
  // SWITCH-NEXT: 1: ^bb2
  %0 = scf.index_switch %i -> i32
  // SWITCH: ^bb1:
  case 0 {
    // SWITCH-NEXT: llvm.br ^bb4(%arg1
    scf.yield %a : i32
  }
  // SWITCH: ^bb2:
  case 1 {
    // SWITCH-NEXT: llvm.br ^bb4(%arg2
    scf.yield %b : i32
  }
  // SWITCH: ^bb3:
  default {
    // SWITCH-NEXT: llvm.br ^bb4(%arg3
    scf.yield %c : i32
  }
  // SWITCH: ^bb4(%[[V:.*]]: i32
  // SWITCH-NEXT: return %[[V]]
  kgen.return %0 : i32
}

// CHECK-LABEL: @arith_select
kgen.func @arith_select(%c: i1, %a: !pop.simd<1, si64>, %b: !pop.simd<1, si64>) -> !pop.simd<1, si64> {
  // CHECK: llvm.select {{.*}} : i1, i64
  %0 = arith.select %c, %a, %b : !pop.simd<1, si64>
  kgen.return %0 : !pop.simd<1, si64>
}
