// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.func @slow_function
kgen.func @slow_function(%arg0: i32) -> !co.routine {
  // CHECK-NEXT: %[[HDL:.*]] = co.handle : i32
  %hdl = co.handle : i32
  // CHECK-NEXT: %[[PROMISE:.*]] = co.promise %[[HDL]] : <struct<(i32)>>
  %promise = co.promise %hdl : <struct<(i32)>>
  // CHECK-NEXT: kgen.struct.gep %[[PROMISE]][0] : <struct<(i32)>>
  %res0 = kgen.struct.gep %promise[0] : <struct<(i32)>>
  kgen.return %hdl : !co.routine
}

// CHECK-LABEL: kgen.func @async_coroutine
kgen.func @async_coroutine(%arg0: i32) -> !co.routine {
  // CHECK: %[[HDL:.*]] = co.handle : i32
  %curHdl = co.handle : i32
  // CHECK: %[[CALLEE_HDL:.*]] = kgen.call @slow_function
  %calleeHdl = kgen.call @slow_function(%arg0) : (i32) -> !co.routine
  // CHECK-NEXT: co.suspend -> %hdl {
  co.suspend -> %hdl {
    // CHECK-NEXT: co.resume %[[CALLEE_HDL]]
    co.resume %calleeHdl
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
  // CHECK: co.execute
  co.execute : i32 {
    // CHECK-NEXT: kgen.unreachable
    kgen.unreachable
  }
  // CHECK: kgen.return %[[HDL]]
  kgen.return %coroHdl : !co.routine
}
