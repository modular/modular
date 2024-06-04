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
