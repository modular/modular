// RUN: kgen-opt %s -lower-kgen-to-llvm=index-bitwidth=64 -lower-scf-to-llvm=index-bitwidth=64 -canonicalize | FileCheck %s

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
