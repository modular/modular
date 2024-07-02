// RUN: kgen-opt -allow-unregistered-dialect -lower-arg-conventions -verify-parameters %s | FileCheck %s

// CHECK-LABEL: kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow)
kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32 borrow) -> si32 {
  // CHECK: kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow)
  %1 = kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32 borrow) -> si32
  kgen.return %1 : si32
}

// CHECK-LABEL: kgen.func @lower_args(
kgen.func @lower_args(
  // CHECK-SAME: %arg0: index owned,
  // CHECK-SAME: %arg1: !kgen.struct<(index, index)> borrow,
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
  // CHECK-SAME: index owned,
  // CHECK-SAME: !kgen.struct<(index, index)> borrow,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call @lower_args(%0, %1, %2, %0) : !lower_args_sig

  // CHECK-DAG: %[[VAL0:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-DAG: %[[VAL1:.*]] = pop.load %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: kgen.call_indirect %arg0(%[[VAL0]], %[[VAL1]], %[[P2]], %[[P0]]) : (
  // CHECK-SAME: index owned,
  // CHECK-SAME: !kgen.struct<(index, index)> borrow,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call_indirect %arg0(%0, %1, %2, %0) : !lower_args_sig
  kgen.return
}

// CHECK-LABEL: kgen.func @byref_res(%arg0: index owned) -> index {
kgen.func @byref_res(%arg0: index owned, %__result__: !kgen.pointer<index> byref_result) -> !kgen.none {
  // CHECK-NEXT: %[[P0:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: "somehow.populate"(%[[P0]]) : (!kgen.pointer<index>) -> ()
  "somehow.populate"(%__result__) : (!kgen.pointer<index>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: %[[RES:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return %[[RES]] : index
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @byref_res_reg_passable(%arg0: index owned) -> !kgen.struct<(index, index)> {
kgen.func @byref_res_reg_passable(%arg0: index owned, %__result__: !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none {
  // CHECK-NEXT: %[[P0:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-NEXT: "somehow.populate"(%[[P0]]) : (!kgen.pointer<struct<(index, index)>>) -> ()
  "somehow.populate"(%__result__) : (!kgen.pointer<struct<(index, index)>>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: %[[RES:.*]] = pop.load %[[P0]] : !kgen.pointer<struct<(index, index)>>
  // CHECK-NEXT: kgen.return %[[RES]] : !kgen.struct<(index, index)>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @byref_res_mem_only
// CHECK-SAME: -> !kgen.none {
kgen.func @byref_res_mem_only(%arg0: index owned, %__result__: !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none {
  // CHECK-NEXT: "somehow.populate"
  "somehow.populate"(%__result__) : (!kgen.pointer<struct<(index, index) memoryOnly>>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: kgen.return %{{.*}} : !kgen.none
  kgen.return %none : !kgen.none
}

!byref_res_sig = !kgen.signature<(index owned, !kgen.pointer<index> byref_result) -> !kgen.none>
!byref_res_reg_passable_sig = !kgen.signature<(index owned, !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none>
!byref_res_mem_only_sig = !kgen.signature<(index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none>

// CHECK-LABEL: kgen.func @test_lower_res
kgen.func @test_lower_res(%arg0: !byref_res_sig, %arg1: !byref_res_reg_passable_sig, %arg2: !byref_res_mem_only_sig, %arg3: index) {
  // CHECK-DAG: %[[P0:.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index
  // CHECK-DAG: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  %1 = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-DAG: %[[P2:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
  %2 = pop.stack_allocation 1 x struct<(index, index) memoryOnly>

  // CHECK: %[[RES0:.*]] = kgen.call @byref_res(%arg3) : (index owned) -> index
  // CHECK-NEXT: pop.store %[[RES0]], %[[P0]] : !kgen.pointer<index>
  kgen.call @byref_res(%arg3, %0) : !byref_res_sig
  // CHECK: %[[RES1:.*]] = kgen.call @byref_res_reg_passable(%arg3) : (index owned) -> !kgen.struct<(index, index)>
  // CHECK-NEXT: pop.store %[[RES1]], %[[P1]] : !kgen.pointer<struct<(index, index)>>
  kgen.call @byref_res_reg_passable(%arg3, %1) : !byref_res_reg_passable_sig
  // CHECK: kgen.call @byref_res_mem_only(%arg3, %2) : (index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none
  // CHECK-NOT: pop.store
  kgen.call @byref_res_mem_only(%arg3, %2) : !byref_res_mem_only_sig

  // CHECK: %[[RES0:.*]] = kgen.call_indirect %arg0(%arg3) : (index owned) -> index
  // CHECK-NEXT: pop.store %[[RES0]], %[[P0]] : !kgen.pointer<index>
  kgen.call_indirect %arg0(%arg3, %0) : !byref_res_sig
  // CHECK: %[[RES1:.*]] = kgen.call_indirect %arg1(%arg3) : (index owned) -> !kgen.struct<(index, index)>
  // CHECK-NEXT: pop.store %[[RES1]], %[[P1]] : !kgen.pointer<struct<(index, index)>>
  kgen.call_indirect %arg1(%arg3, %1) : !byref_res_reg_passable_sig
  // kgen.call_indirect %arg2(%[[P2]], %arg3) : (index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none
  // CHECK-NOT: pop.store
  kgen.call_indirect %arg2(%arg3, %2) : !byref_res_mem_only_sig

  kgen.return
}

!Error = !kgen.struct<(f32)>

// CHECK-LABEL: kgen.func @byref_throws() throws -> !kgen.variant<struct<(f32)>, index>
kgen.func @byref_throws(
  %__error__: !kgen.pointer<!Error> byref_error,
  %__result__: !kgen.pointer<index> byref_result
) throws -> i1 {
  // CHECK: %[[VALUE:.*]] = pop.stack_allocation 1 x index
  // CHECK: %[[ERROR:.*]] = pop.stack_allocation 1 x struct<(f32)>

  // CHECK: %[[COND:.*]] = kgen.param.constant: i1 = <?>
  %0 = kgen.param.constant: i1 = <?>

  // CHECK: hlcf.if %[[COND]]
  hlcf.if %0 {
    %1 = kgen.param.constant: i1 = <1>
    // CHECK: [[ERR:%.*]] = pop.load %[[ERROR]]
    // CHECK-NEXT: [[RESULT:%.*]] = kgen.variant.create [[ERR]], 0
    // CHECK-NEXT: return [[RESULT]]
    kgen.return %1 : i1
  } else {
    %2 = kgen.param.constant: i1 = <0>
    // CHECK: [[VAL:%.*]] = pop.load %[[VALUE]]
    // CHECK-NEXT: [[RESULT:%.*]] = kgen.variant.create [[VAL]], 1
    // CHECK-NEXT: return [[RESULT]]
    kgen.return %2 : i1
  }

  // CHECK:      %[[RES:.*]] = hlcf.if %[[COND]] -> !kgen.variant
  // CHECK-NEXT:   [[ERR:%.*]] = pop.load %[[ERROR]]
  // CHECK-NEXT:   [[RESULT:%.*]] = kgen.variant.create [[ERR]], 0
  // CHECK-NEXT:   hlcf.yield [[RESULT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = pop.load %[[VALUE]]
  // CHECK-NEXT:   [[RESULT:%.*]] = kgen.variant.create [[VAL]], 1
  // CHECK-NEXT:   hlcf.yield [[RESULT]]

  // CHECK: kgen.return %[[RES]]
  kgen.return %0 : i1
}

!byref_throws_sig = !kgen.signature<(
  !kgen.pointer<!Error> byref_error, !kgen.pointer<index> byref_result
) throws -> i1>

// CHECK-LABEL: kgen.func @test_byref_throws(
kgen.func @test_byref_throws(%arg0: !byref_throws_sig) {
  // CHECK-NEXT: [[RESULT:%.*]] = pop.stack_allocation 1 x index
  %__result__ = pop.stack_allocation 1 x index
  // CHECK-NEXT: [[ERROR:%.*]] = pop.stack_allocation 1 x struct<(f32)>
  %__error__ = pop.stack_allocation 1 x !Error

  // CHECK: [[RES:%.*]] = kgen.call @byref_throws()
  // CHECK-NEXT: [[IS_ERR:%.*]] = kgen.variant.is [[RES]], 0
  // CHECK-NEXT: hlcf.if [[IS_ERR]] {
  // CHECK-NEXT:   [[ERR:%.*]] = kgen.variant.take [[RES]], 0
  // CHECK-NEXT:   store [[ERR]], [[ERROR]]
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = kgen.variant.take [[RES]], 1
  // CHECK-NEXT:   store [[VAL]], [[RESULT]]
  // CHECK-NEXT:   yield
  %res1 = kgen.call @byref_throws(%__error__, %__result__) : (
    !kgen.pointer<!Error> byref_error, !kgen.pointer<index> byref_result
  ) throws -> i1
  "handle.error"(%res1) : (i1) -> ()
  "use.result"(%__result__) : (!kgen.pointer<index>) -> ()


  // CHECK: [[RES:%.*]] = kgen.call_indirect %arg0()
  // CHECK-NEXT: [[IS_ERR:%.*]] = kgen.variant.is [[RES]], 0
  // CHECK-NEXT: hlcf.if [[IS_ERR]] {
  // CHECK-NEXT:   [[ERR:%.*]] = kgen.variant.take [[RES]], 0
  // CHECK-NEXT:   store [[ERR]], [[ERROR]]
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = kgen.variant.take [[RES]], 1
  // CHECK-NEXT:   store [[VAL]], [[RESULT]]
  // CHECK-NEXT:   yield
  %res2 = kgen.call_indirect %arg0(%__error__, %__result__) : !byref_throws_sig
  "handle.error"(%res2) : (i1) -> ()
  "use.result"(%__result__) : (!kgen.pointer<index>) -> ()
}

// CHECK-LABEL: @self_result_and_arg
// CHECK-SAME: (%arg0: !kgen.struct<()> borrow, %arg1: i8 borrow) -> !kgen.struct<()>
kgen.func @self_result_and_arg(%arg1: !kgen.pointer<struct<()>> borrow_in_mem,
                               %arg2: i8 borrow,
                               %arg0: !kgen.pointer<struct<()>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @call_it_self_result_and_arg
// CHECK-SAME: %arg0: !kgen.struct<()>
kgen.func @call_it_self_result_and_arg(%arg0: !kgen.pointer<struct<()>> borrow_in_mem) -> !kgen.none {
  %0 = pop.stack_allocation 1 x struct<()>
  // CHECK: %[[CST:.*]] = kgen.param.constant: i8 = <4>
  %1 = kgen.param.constant: i8 = <4>
  // CHECK: call @self_result_and_arg(%{{.*}}, %[[CST]]) : (!kgen.struct<()> borrow, i8 borrow) -> !kgen.struct<()>
  %2 = kgen.call @self_result_and_arg(%arg0, %1, %0) : (!kgen.pointer<struct<()>> borrow_in_mem, i8 borrow, !kgen.pointer<struct<()>> byref_result) -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @reg__init__() -> !kgen.struct<()>
kgen.func @reg__init__(%arg0: !kgen.pointer<struct<()>> init_self) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @init_a_reg_type
kgen.func @init_a_reg_type() {
  %0 = pop.stack_allocation 1 x struct<()>
  // CHECK: call @reg__init__() : () -> !kgen.struct<()>
  kgen.call @reg__init__(%0) : (!kgen.pointer<struct<()>> init_self) -> !kgen.none
  kgen.return
}

// CHECK-LABEL: kgen.func @unreachable_byref_result() -> index
kgen.func @unreachable_byref_result(%arg0: !kgen.pointer<index> byref_result) -> !kgen.none {
  // CHECK: loop
  hlcf.loop {
    %none = kgen.param.constant: none = <#kgen.none>
    // CHECK: [[R:%.*]] = pop.load
    // CHECK-NEXT: return [[R]]
    kgen.return %none : !kgen.none
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: unreachable
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @byref_error
// CHECK-SAME: (%arg0: !kgen.signature<(index, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> (i1, f16)>
// CHECK-SAME:  %arg1: !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> (i1, f16)
kgen.func @byref_error(
    %f: !kgen.signature<(index, !kgen.pointer<f16> byref_error, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> i1>,
    %err: !kgen.pointer<f16> byref_error,
    %result: !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> i1 {
  // CHECK-NEXT: [[F_ERR:%.*]] = pop.stack_allocation 1 x f16
  // CHECK-NEXT: %idx0
  %idx0 = index.constant 0

  // CHECK-NEXT: [[ERR:%.*]] = pop.stack_allocation 1 x f16
  // CHECK-NEXT: [[VAL:%.*]] = pop.stack_allocation 1 x struct<(i16) memoryOnly>
  %0 = pop.stack_allocation 1 x f16
  %1 = pop.stack_allocation 1 x struct<(i16) memoryOnly>
  // CHECK-NEXT: [[R:%.*]]:2 = kgen.call_indirect %arg0(%idx0, [[VAL]])
  %2 = kgen.call_indirect %f(%idx0, %0, %1) : (index, !kgen.pointer<f16> byref_error, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> i1
  // CHECK-NEXT: hlcf.if [[R]]#0 {
  // CHECK-NEXT:   pop.store [[R]]#1, [[ERR]]
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }

  // CHECK-NEXT: [[ERR_RES:%.*]] = pop.load [[F_ERR]]
  // CHECK-NEXT: return [[R]]#0, [[ERR_RES]]
  kgen.return %2 : i1
}

// CHECK-LABEL: kgen.func @initself_value
// CHECK-SAME: (%arg0: !kgen.signature<(index, !kgen.pointer<struct<(f16) memoryOnly>> byref_error) throws -> (i1, i16)>
// CHECK-SAME:  %arg1: !kgen.pointer<struct<(f16) memoryOnly>> byref_error) throws -> (i1, i16)
kgen.func @initself_value(
    %self: !kgen.pointer<i16> init_self,
    %f: !kgen.signature<(!kgen.pointer<i16> init_self, index, !kgen.pointer<struct<(f16) memoryOnly>> byref_error) throws -> i1>,
    %err: !kgen.pointer<struct<(f16) memoryOnly>> byref_error) throws -> i1 {
  // CHECK-NEXT: [[F_RESULT:%.*]] = pop.stack_allocation 1 x i16
  // CHECK-NEXT: %idx0
  %idx0 = index.constant 0

  // CHECK-NEXT: [[ERR:%.*]] = pop.stack_allocation 1 x struct<(f16) memoryOnly>
  // CHECK-NEXT: [[VAL:%.*]] = pop.stack_allocation 1 x i16
  %0 = pop.stack_allocation 1 x struct<(f16) memoryOnly>
  %1 = pop.stack_allocation 1 x i16
  // CHECK-NEXT: [[R:%.*]]:2 = kgen.call_indirect %arg0(%idx0, [[ERR]])
  %2 = kgen.call_indirect %f(%1, %idx0, %0) : (!kgen.pointer<i16> init_self, index, !kgen.pointer<struct<(f16) memoryOnly>> byref_error) throws -> i1
  // CHECK-NEXT: hlcf.if [[R]]#0 {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   pop.store [[R]]#1, [[VAL]]
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }

  // CHECK-NEXT: [[VAL_RES:%.*]] = pop.load [[F_RESULT]]
  // CHECK-NEXT: return [[R]]#0, [[VAL_RES]]
  kgen.return %2 : i1
}

// CHECK-LABEL: @dont_alter_async_results(%arg0: index borrow, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async
kgen.func @dont_alter_async_results(%arg0: !kgen.pointer<index> borrow_in_mem, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async {
  kgen.return
}

// CHECK-LABEL: kgen.func @two_call_indirect
kgen.func @two_call_indirect(%arg0: !kgen.signature<(!kgen.pointer<index> byref_result) -> !kgen.none>) {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  // CHECK: kgen.call_indirect %arg0() : () -> index
  kgen.call_indirect %arg0(%0) : (!kgen.pointer<index> byref_result) -> !kgen.none
  // CHECK: kgen.call_indirect %arg0() : () -> index
  kgen.call_indirect %arg0(%1) : (!kgen.pointer<index> byref_result) -> !kgen.none
  kgen.return
}
