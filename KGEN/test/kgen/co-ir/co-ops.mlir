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
