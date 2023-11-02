// RUN: kgen-opt -allow-unregistered-dialect -lower-input-conventions -verify-parameters %s | FileCheck %s

// CHECK-LABEL: kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow)
kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow) -> si32{
  // CHECK: kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow)
  %1 = kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow) -> si32
  kgen.return %1 : si32
}

// CHECK-LABEL: kgen.func @lower_args(
kgen.func @lower_args(
  // CHECK-SAME: %arg0: !kgen.pointer<index>,
  // CHECK-SAME: %arg1: !kgen.pointer<index>,
  // CHECK-SAME: %arg2: !kgen.pointer<index> owned)
  %arg0: !kgen.pointer<index> owned_in_mem,
  %arg1: !kgen.pointer<index> borrow_in_mem,
  %arg2: !kgen.pointer<index> owned
) {
  "some.use"(%arg0) : (!kgen.pointer<index>) -> ()
  kgen.return
}

// CHECK: kgen.func @test_lower_args
kgen.func @test_lower_args() {
  %0 = pop.stack_allocation 1 x index
  // CHECK: kgen.call @lower_args({{.*}}) : (!kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index> owned) -> ()
  kgen.call @lower_args(%0, %0, %0) : (!kgen.pointer<index> owned_in_mem, !kgen.pointer<index> borrow_in_mem, !kgen.pointer<index> owned) -> ()
  kgen.return
}
