// RUN: kgen-opt %s -allow-unregistered-dialect -lower-kgen-to-llvm=index-bitwidth=64 -lower-scf-to-llvm=index-bitwidth=64 -canonicalize | FileCheck %s

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
kgen.func @while(%init: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: llvm.br ^bb1(%[[INIT]]
  %result = scf.while (%v = %init) : (!pop.scalar<f32>) -> !pop.scalar<f32> {
    // CHECK: ^bb1(%[[V:.*]]: f32
    // CHECK: llvm.cond_br %{{.*}}, ^bb2(%[[V]] : f32), ^bb3
    %condition = "cond"(%v) : (!pop.scalar<f32>) -> i1
    scf.condition(%condition) %v : !pop.scalar<f32>
  } do {
  // CHECK: ^bb2(%[[U:.*]]: f32
  ^bb0(%u: !pop.scalar<f32>):
    // CHECK: llvm.br ^bb1(
    %next = "next"(%u) : (!pop.scalar<f32>) -> !pop.scalar<f32>
    scf.yield %next : !pop.scalar<f32>
  }
  // CHECK: ^bb3:
  // CHECK: return %[[V]]
  kgen.return %result : !pop.scalar<f32>
}

// CHECK-LABEL: @arith_select
kgen.func @arith_select(%c: i1, %a: !pop.scalar<si64>, %b: !pop.scalar<si64>) -> !pop.scalar<si64> {
  // CHECK: llvm.select {{.*}} : i1, i64
  %0 = arith.select %c, %a, %b : !pop.scalar<si64>
  kgen.return %0 : !pop.scalar<si64>
}
