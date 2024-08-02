// RUN: kgen-opt -lower-calling-conventions -allow-unregistered-dialect %s | FileCheck %s

//===----------------------------------------------------------------------===//
// `!kgen.none` lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.func @none_func() {
kgen.func @none_func() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  // CHECK: kgen.return
  // CHECK-NOT: !kgen.none
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @none_func_other_results
// CHECK-SAME: -> (i32, i64)
kgen.func @none_func_other_results(%arg0: i32, %arg1: i64) -> (i32, !kgen.none, i64) {
  %none = kgen.param.constant: none = <#kgen.none>
  // CHECK: return %arg0, %arg1
  kgen.return %arg0, %none, %arg1 : i32, !kgen.none, i64
}

// CHECK-LABEL: kgen.func @none_stage_closure
// CHECK-SAME: -> !kgen.signature<() -> ()>
kgen.func @none_stage_closure() -> !kgen.signature<() -> !kgen.none> {
  // CHECK: kgen.stage_closure = () {
  %0 = kgen.stage_closure = () -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    // CHECK: return
    // CHECK-NOT: !kgen.none
    kgen.return %none : !kgen.none
  }
  // CHECK: return %0 : !kgen.signature<() -> ()>
  kgen.return %0 : !kgen.signature<() -> !kgen.none>
}

// CHECK-LABEL: @early_return_loop
kgen.func @early_return_loop() -> !kgen.none {
  %0 = kgen.param.constant:none = <#kgen.none>
  hlcf.loop {
    kgen.return %0 : !kgen.none
  }
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @if_none
kgen.func @if_none(%arg0: i1, %arg1: i32) {
  %none = kgen.param.constant: none = <#kgen.none>
  // CHECK: %0 = hlcf.if %arg0 -> i32
  %0:2 = hlcf.if %arg0 -> !kgen.none, i32 {
    // CHECK: hlcf.yield %arg1
    hlcf.yield %none, %arg1 : !kgen.none, i32
  } else {
    kgen.unreachable
  }
  // CHECK: "use.op"(%none{{.*}}, %0)
  "use.op"(%0#0, %0#1) : (!kgen.none, i32) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @loop_none
kgen.func @loop_none() {
  %none = kgen.param.constant: none = <#kgen.none>
  // CHECK: hlcf.loop {
  hlcf.loop () -> !kgen.none {
    // CHECK: hlcf.break
    // CHECK-NOT: !kgen.none
    hlcf.break %none : !kgen.none
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @call_none
kgen.func @call_none() {
  // CHECK: %0:2 = kgen.call @none_extern_func() : () -> (i32, i64)
  %0:3 = kgen.call @none_extern_func() : () -> (i32, !kgen.none, i64)
  // CHECK: "use.op"(%0#0, %none, %0#1)
  "use.op"(%0#0, %0#1, %0#2) : (i32, !kgen.none, i64) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call_indirect
// CHECK-SAME: %arg0: !kgen.signature<() -> ()>
kgen.func @call_indirect(%arg0: !kgen.signature<() -> !kgen.none>) {
  // CHECK: kgen.call_indirect %arg0() : () -> ()
  %0 = kgen.call_indirect %arg0() : () -> !kgen.none
  kgen.return
}

// CHECK-LABEL: kgen.func @async_signature
// CHECK-SAME: !kgen.signature<() async -> ()>
kgen.func @async_signature(%arg0: !kgen.signature<() async -> !kgen.none>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @co_get_results
kgen.func @co_get_results(%arg0: !co.routine) {
  // CHECK-NEXT: %0 = co.get_results %arg0 : i32
  %0:2 = co.get_results %arg0 : !kgen.none, i32
  kgen.return
}

// CHECK-LABEL: kgen.func @co_await
kgen.func @co_await(%arg0: !co.routine) -> !kgen.none {
  %pointer = kgen.param.constant: pointer<index> = <0>
  // CHECK: co.await %arg0, %pointer, %pointer : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> ()
  %0 = co.await %arg0, %pointer, %pointer : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> !kgen.none
  kgen.return %0 : !kgen.none
}

//===----------------------------------------------------------------------===//
// `!kgen.pack` lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @lower_pack
// CHECK-SAME: %arg0: !kgen.struct<(i32, i64)>
// CHECK-SAME: -> !kgen.struct<(i32, i64)>
kgen.func @lower_pack(%arg0: !kgen.pack<[i32, i64]>) -> !kgen.pack<[i32, i64]> {
  kgen.return %arg0 : !kgen.pack<[i32, i64]>
}

// CHECK-LABEL: @pack_create
kgen.func @pack_create(%arg0: i32, %arg1: i64) -> !kgen.pack<[i32, i64]> {
  // CHECK-NEXT: %0 = kgen.struct.create(%arg0, %arg1) : !kgen.struct<(i32, i64)>
  %0 = kgen.pack.create(%arg0, %arg1) : !kgen.pack<[i32, i64]>
  kgen.return %0 : !kgen.pack<[i32, i64]>
}

// CHECK-LABEL: @pack_extract
kgen.func @pack_extract(%arg0: !kgen.pack<[i32, i64]>) -> i64 {
  // CHECK-NEXT: %0 = kgen.struct.extract %arg0[1] : !kgen.struct<(i32, i64)>
  %0 = kgen.pack.extract %arg0[1] : <[i32, i64]>
  kgen.return %0 : i64
}

// CHECK-LABEL: @pack_gep
kgen.func @pack_gep(%arg0: !kgen.pointer<!kgen.pack<[i32, i64]>>) -> !kgen.pointer<i32> {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(i32, i64)>>
  %0 = kgen.pack.gep %arg0[0] : <!kgen.pack<[i32, i64]>>
  kgen.return %0 : !kgen.pointer<i32>
}

// CHECK-LABEL: @pack_size
kgen.func @pack_size(%arg0: !kgen.pack<[i32, i64]>) -> index {
  // CHECK-NEXT: %index2 = kgen.param.constant = <2>
  %0 = kgen.pack.size %arg0 : <[i32, i64]>
  // CHECK-NEXT: return %index2
  kgen.return %0 : index
}

// CHECK-LABEL: @pack_load
kgen.func @pack_load(%arg0: !kgen.pack<[pointer<i32>, pointer<i64>]>) -> !kgen.pack<[i32, i64]> {
  // CHECK-NEXT: [[PTR0:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<i32>, pointer<i64>)>
  // CHECK-NEXT: [[EL0:%.*]] = pop.load [[PTR0]]
  // CHECK-NEXT: [[PTR1:%.*]] = kgen.struct.extract %arg0[1] : !kgen.struct<(pointer<i32>, pointer<i64>)>
  // CHECK-NEXT: [[EL1:%.*]] = pop.load [[PTR1]]
  // CHECK-NEXT: [[RESULT:%.*]] = kgen.struct.create([[EL0]], [[EL1]]) : !kgen.struct<(i32, i64)>
  %0 = kgen.pack.load %arg0 : <[pointer<i32>, pointer<i64>]>
  // CHECK-NEXT: return [[RESULT]]
  kgen.return %0 : !kgen.pack<[i32, i64]>
}

// CHECK-LABEL: @nested_pack_attr
kgen.func @nested_pack_attr() {
  // CHECK-NEXT: constant: struct<(struct<()>, struct<(i32, i64)>)> = <{ { }, { 1, 2 } }>
  kgen.param.constant: !kgen.pack<[!kgen.pack<[]>, !kgen.pack<[i32, i64]>]> = <#kgen.pack<#kgen.pack<>, #kgen.pack<1, 2>>>
  kgen.return
}

kgen.func @memtype__moveinit__(%arg0: !kgen.pointer<struct<(index) memoryOnly>> init_self, %arg1: !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}
// CHECK-LABEL kgen.func @memtype_create_reg_stub
kgen.func @memtype_create_reg_stub() -> !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> {
  // CHECK: kgen.create_closure[(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> (): @memtype__moveinit__]()
  %0 = kgen.create_reg_stub [(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none: @memtype__moveinit__] : <(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
  kgen.return %0 : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
}

kgen.func @regtype__moveinit__(%arg0: index owned) -> index {
  kgen.return %arg0 : index
}
// CHECK-LABEL: kgen.func @regtype_create_reg_stub
kgen.func @regtype_create_reg_stub() -> !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none> {
  // CHECK: kgen.stage_closure = ([[ARG0:%.*]]: !kgen.pointer<struct<(index) memoryOnly>> init_self, [[ARG1:%.*]]: !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) {
  // CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[ARG0]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<index>
  // CHECK-NEXT: [[V2:%.*]] = pop.pointer.bitcast [[ARG1]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = pop.load [[V2]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V4:%.*]] = kgen.call @regtype__moveinit__([[V3]]) : (index owned) -> index
  // CHECK-NEXT: pop.store [[V4]], [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return
  %0 = kgen.create_reg_stub [(index owned) -> index: @regtype__moveinit__] : <(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
  kgen.return %0 : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
}

kgen.func @mixtypes_fun(%arg0: !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, %arg1: !kgen.pointer<none> borrow, %arg2: !pop.scalar<si16> borrow) -> index {
  %0 = kgen.param.constant = <1>
  kgen.return %0 : index
}
// CHECK-LABEL: kgen.func @mixtypes_create_reg_stub
kgen.func @mixtypes_create_reg_stub() -> !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none> borrow, !pop.scalar<si16> borrow) -> !kgen.none> {
  // CHECK: kgen.stage_closure = ([[ARG0:%.*]]: !kgen.pointer<struct<(index) memoryOnly>> init_self, [[ARG1:%.*]]: !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, [[ARG2:%.*]]: !kgen.pointer<none>, [[ARG3:%.*]]: !pop.scalar<si16>) {
  // CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[ARG0]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<index>
  // CHECK-NEXT: [[V2:%.*]] = kgen.call @mixtypes_fun([[ARG1]], [[ARG2]], [[ARG3]]) : (!kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none>, !pop.scalar<si16>) -> index
  // CHECK-NEXT: pop.store [[V2]], [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return
  %0 = kgen.create_reg_stub [(!kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none> borrow, !pop.scalar<si16> borrow) -> index: @mixtypes_fun] : <(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none> borrow, !pop.scalar<si16> borrow) -> !kgen.none>
  kgen.return %0 : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> init_self, !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none> borrow, !pop.scalar<si16> borrow) -> !kgen.none>
}

kgen.func @byrefresult_fun(%arg0: index, %arg1: !kgen.pointer<none>) -> !pop.scalar<si32> {
  %0 = kgen.param.constant: scalar<si32> = <42>
  kgen.return %0 : !pop.scalar<si32>
}
// CHECK-LABEL: kgen.func @byrefresult_create_reg_stub
kgen.func @byrefresult_create_reg_stub() -> !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none>, !kgen.pointer<struct<(scalar<si32>) memoryOnly>> byref_result) -> !kgen.none> {
  // CHECK: kgen.stage_closure = ([[ARG0:%.*]]: !kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, [[ARG1:%.*]]: !kgen.pointer<none>, [[ARG2:%.*]]: !kgen.pointer<struct<(scalar<si32>) memoryOnly>> byref_result) {
  // CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[ARG0]] : !kgen.pointer<struct<(index) memoryOnly>> to !kgen.pointer<index>
  // CHECK-NEXT: [[V2:%.*]] = pop.load [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = pop.pointer.bitcast [[ARG2]] : !kgen.pointer<struct<(scalar<si32>) memoryOnly>> to !kgen.pointer<scalar<si32>>
  // CHECK-NEXT: [[V4:%.*]] = kgen.call @byrefresult_fun([[V2]], [[ARG1]]) : (index, !kgen.pointer<none>) -> !pop.scalar<si32>
  // CHECK-NEXT: pop.store [[V4]], [[V3]] : !kgen.pointer<scalar<si32>>
  // CHECK-NEXT: kgen.return
  %0 = kgen.create_reg_stub [(index, !kgen.pointer<none>) -> !pop.scalar<si32>: @byrefresult_fun] : <(!kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none>, !kgen.pointer<struct<(scalar<si32>) memoryOnly>> byref_result) -> !kgen.none>
  kgen.return %0 : !kgen.signature<(!kgen.pointer<struct<(index) memoryOnly>> borrow_in_mem, !kgen.pointer<none>, !kgen.pointer<struct<(scalar<si32>) memoryOnly>> byref_result) -> !kgen.none>
}

kgen.func @noargs_fun() -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}
// CHECK-LABEL kgen.func @noargs_create_reg_stub
kgen.func @noargs_create_reg_stub() -> !kgen.signature<() -> !kgen.none> {
  // CHECK: kgen.create_closure[() -> (): @noargs_create_reg_stub]()
  %0 = kgen.create_reg_stub [() -> !kgen.none: @noargs_create_reg_stub] : <() -> !kgen.none>
  kgen.return %0 : !kgen.signature<() -> !kgen.none>
}

// CHECK-LABEL: @lower_variants
kgen.func @lower_variants(%arg0: i64) {
  // CHECK-NEXT: kgen.param.constant: struct<(union<i32, i64>, scalar<ui8>)> = <{ {:i32 44}, 0 }>
  kgen.param.constant: variant<i32, i64> = <#kgen.variant<:i32 44, 0>>

  // CHECK-NEXT: [[UNION:%.*]] = pop.union.wrap %arg0 : i64 as <i32, i64>
  // CHECK-NEXT: [[DISCR:%.*]] = kgen.param.constant: scalar<ui8> = <1>
  // CHECK-NEXT: [[VARIANT:%.*]] = kgen.struct.create([[UNION]], [[DISCR]]) : !kgen.struct<(union<i32, i64>, scalar<ui8>)>
  %0 = kgen.variant.create %arg0, 1 : <i32, i64>

  // CHECK-NEXT: [[DISCR:%.*]] = kgen.struct.extract [[VARIANT]][1]
  // CHECK-NEXT: [[ZERO:%.*]] = kgen.param.constant: scalar<ui8> = <0>
  // CHECK-NEXT: [[EQ:%.*]] = pop.cmp eq([[DISCR]], [[ZERO]])
  // CHECK-NEXT: pop.cast_to_builtin [[EQ]] : !pop.scalar<bool> to i1
  %1 = kgen.variant.is %0, 0 : <i32, i64>

  // CHECK-NEXT: [[UNION:%.*]] = kgen.struct.extract [[VARIANT]][0]
  // CHECK-NEXT: pop.union.unwrap [[UNION]] : <i32, i64> as i32
  %2 = kgen.variant.get %0, 0 : <i32, i64>

  // CHECK-NEXT: [[ALLOC:%.*]] = pop.stack_allocation
  %3 = pop.stack_allocation 1 x variant<i32, i64>

  // CHECK-NEXT: [[UNION:%.*]] = kgen.struct.gep [[ALLOC]][0]
  // CHECK-NEXT: pop.union.bitcast [[UNION]] : <union<i32, i64>> as <i64>
  %4 = pop.variant.bitcast %3, <1> : <variant<i32, i64>> as <i64>

  // CHECK-NEXT: kgen.struct.gep [[ALLOC]][1]
  %5 = pop.variant.discr_gep %3 : <variant<i32, i64>> as <scalar<ui8>>
  kgen.return
}
