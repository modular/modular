// RUN: kgen-opt -allow-unregistered-dialect -lower-input-conventions -verify-parameters %s -o %t
// RUN: cat %t | FileCheck %s
// RUN: kgen-opt -allow-unregistered-dialect -mem-2-reg %t | FileCheck %s --check-prefix=MEM-2-REG

// CHECK-LABEL: kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow)
kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow) -> si32 {
  // CHECK: kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow)
  %1 = kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow) -> si32
  kgen.return %1 : si32
}

// CHECK-LABEL: kgen.func @lower_args(
kgen.func @lower_args(
  // CHECK-SAME: %arg0: index,
  // CHECK-SAME: %arg1: !kgen.struct<(index, index)>,
  // CHECK-SAME: %arg2: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  // CHECK-SAME: %arg3: !kgen.pointer<index> owned
  %arg0: !kgen.pointer<index> owned_in_mem,
  %arg1: !kgen.pointer<struct<(index, index)>> borrow_in_mem,
  %arg2: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  %arg3: !kgen.pointer<index> owned
) {
  // CHECK: %[[P0:.*]] = pop.stack_allocation 1 x index
  // CHECK: pop.store %arg0, %[[P0]] : !kgen.pointer<index>
  // CHECK: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK: pop.store %arg1, %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: "some.use"(%[[P0]], %[[P1]])
  "some.use"(%arg0, %arg1) : (!kgen.pointer<index>, !kgen.pointer<struct<(index, index)>>) -> ()
  kgen.return
}

!lower_args_sig = !kgen.signature<(
  !kgen.pointer<index> owned_in_mem,
  !kgen.pointer<struct<(index, index)>> borrow_in_mem,
  !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  !kgen.pointer<index> owned
) -> ()>

// CHECK: kgen.func @test_lower_args
kgen.func @test_lower_args(%arg0: !lower_args_sig) {
  // CHECK-DAG: %[[P0:.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index
  // CHECK-DAG: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  %1 = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-DAG: %[[P2:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
  %2 = pop.stack_allocation 1 x struct<(index, index) memoryOnly>

  // CHECK-DAG: %[[VAL0:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-DAG: %[[VAL1:.*]] = pop.load %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: kgen.call @lower_args(%[[VAL0]], %[[VAL1]], %[[P2]], %[[P0]]) : (
  // CHECK-SAME: index,
  // CHECK-SAME: !kgen.struct<(index, index)>,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call @lower_args(%0, %1, %2, %0) : !lower_args_sig

  // CHECK-DAG: %[[VAL0:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-DAG: %[[VAL1:.*]] = pop.load %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: kgen.call_signature %arg0(%[[VAL0]], %[[VAL1]], %[[P2]], %[[P0]]) : (
  // CHECK-SAME: index,
  // CHECK-SAME: !kgen.struct<(index, index)>,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call_signature %arg0(%0, %1, %2, %0) : !lower_args_sig
  kgen.return
}


// MEM-2-REG-LABEL: kgen.func @lower_args_mem_2_reg(%arg0: index, %arg1: !kgen.struct<(index, index)>) {
kgen.func @lower_args_mem_2_reg(
  %arg0: !kgen.pointer<index> owned_in_mem,
  %arg1: !kgen.pointer<struct<(index, index)>> borrow_in_mem
) {
  // MEM-2-REG-NEXT: kgen.call @lower_args_mem_2_reg(%arg0, %arg1) : (index, !kgen.struct<(index, index)>) -> ()
  kgen.call @lower_args_mem_2_reg(%arg0, %arg1) : (
    !kgen.pointer<index> owned_in_mem,
    !kgen.pointer<struct<(index, index)>> borrow_in_mem
  ) -> ()
  kgen.return
}
