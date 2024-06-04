// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.func @slow_function
kgen.func @slow_function(%arg0: i32) -> !co.routine {
  // CHECK-NEXT: %[[HDL:.*]] = co.handle : i32
  %hdl = co.handle : i32
  // CHECK-NEXT: co.set_results %[[HDL]](%arg0) : i32
  co.set_results %hdl(%arg0) : i32
  kgen.return %hdl : !co.routine
}

// CHECK-LABEL: kgen.func @async_coroutine
kgen.func @async_coroutine(%arg0: i32) -> !co.routine {
  // CHECK: %[[HDL:.*]] = co.handle : i32
  %curHdl = co.handle : i32
  // CHECK: %[[CALLEE_HDL:.*]] = kgen.call @slow_function
  %calleeHdl = kgen.call @slow_function(%arg0) : (i32) -> !co.routine
  // CHECK-NEXT: co.suspend (%hdl) {
  co.suspend (%hdl) {
    // CHECK-NEXT: co.resume %[[CALLEE_HDL]] : <(!co.routine) -> ()>
    co.resume %calleeHdl : <(!co.routine) -> ()>
    // CHECK-NEXT: co.suspend.end
    co.suspend.end
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: co.destroy %[[CALLEE_HDL]]
  co.destroy %calleeHdl
  kgen.return %curHdl : !co.routine
}

// CHECK-LABEL: kgen.func @async_execute
kgen.func @async_execute() -> !co.routine {
  // CHECK: [[R0:%.*]] = kgen.param.constant: i32
  %0 = kgen.param.constant: i32 = <3>
  // CHECK: %[[HDL:.*]] = co.execute : i32, i64 {
  %coroHdl = co.execute : i32, i64 {
    // CHECK: [[R1:%.*]] = kgen.param.constant: i64
    %1 = kgen.param.constant: i64 = <5>
    // CHECK: kgen.return [[R0]], [[R1]] : i32, i64
    kgen.return %0, %1 : i32, i64
  }
  // CHECK: co.execute : i32 {
  co.execute : i32 {
    // CHECK-NEXT: kgen.unreachable
    kgen.unreachable
  }
  // CHECK: co.execute : i32 (%arg0: !kgen.pointer<index> byref_result) {
  co.execute : i32 (%arg0: !kgen.pointer<index> byref_result) {
    kgen.unreachable
  }
  // CHECK: co.execute : i32, i64 (%arg0: !kgen.pointer<index> byref_error, %arg1: !kgen.pointer<index> byref_result) {
  co.execute : i32, i64 (%arg0: !kgen.pointer<index> byref_error, %arg1: !kgen.pointer<index> byref_result) {
    kgen.unreachable
  }
  // CHECK: kgen.return %[[HDL]]
  kgen.return %coroHdl : !co.routine
}

kgen.func @async_fn(%arg0: index) async {
  kgen.return
}

// CHECK-LABEL: @call_async_fn
kgen.func @call_async_fn(%arg0: index) -> !co.routine {
  // CHECK-NEXT: co.invoke[(index) async -> (): @async_fn](%arg0)
  %0 = co.invoke[(index) async -> (): @async_fn](%arg0)
  kgen.return %0 : !co.routine
}

kgen.func @throwing_coroutine(%arg0: index, %__error__: !kgen.pointer<index> byref_error, %__result__: !kgen.pointer<index> byref_result) throws|async -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

// CHECK-LABEL: kgen.func @call_throwing_coro
kgen.func @call_throwing_coro(%arg0: index) {
  %size = index.constant 1
  %align = index.constant 8
  // CHECK: %0 = co.invoke[(index, !kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> i1: @throwing_coroutine](%arg0)
  %0 = co.invoke[(index, !kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> i1: @throwing_coroutine](%arg0)
  %1 = pop.aligned_alloc %align, %size : <index>
  %2 = pop.aligned_alloc %align, %size : <index>
  // CHECK: co.set_byref_error_result %0(%1, %2) : <index>, <index>
  co.set_byref_error_result %0(%1, %2) : <index>, <index>
  kgen.return
}
