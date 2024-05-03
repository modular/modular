// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.func @slow_function
kgen.func @slow_function(%arg0: i32) -> !co.routine<() -> i32> {
  // CHECK-NEXT: %[[HDL:.*]] = co.handle : <() -> i32>
  %hdl = co.handle : <() -> i32>
  // CHECK-NEXT: %[[PROMISE:.*]] = co.promise %[[HDL]] : <() -> i32>
  %promise = co.promise %hdl : <() -> i32>
  // CHECK-NEXT: kgen.struct.gep %[[PROMISE]][0] : <struct<(i32)>>
  %res0 = kgen.struct.gep %promise[0] : <struct<(i32)>>
  kgen.return %hdl : !co.routine<() -> i32>
}

// CHECK-LABEL: kgen.func @async_coroutine
kgen.func @async_coroutine(%arg0: i32) -> !co.routine<() -> i32> {
  // CHECK: %[[HDL:.*]] = co.handle : <() -> i32>
  %hdl = co.handle : <() -> i32>
  // CHECK: %[[OPAQUE:.*]] = co.opaque_handle
  %opaque = co.opaque_handle
  // CHECK: %[[CALLEE_HDL:.*]] = kgen.call @slow_function
  %calleeHdl = kgen.call @slow_function(%arg0) : (i32) -> !co.routine<() -> i32>
  // CHECK-NEXT: co.await {
  co.await {
    // CHECK-NEXT: co.resume %[[CALLEE_HDL]] : !co.routine<() -> i32>
    co.resume %calleeHdl : !co.routine<() -> i32>
    // CHECK-NEXT: co.resume %[[OPAQUE]] : !kgen.pointer<i8>
    co.resume %opaque : !kgen.pointer<i8>
    // CHECK-NEXT: co.await.end
    co.await.end
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: co.destroy %[[CALLEE_HDL]] : <() -> i32>
  co.destroy %calleeHdl : <() -> i32>
  kgen.return %hdl : !co.routine<() -> i32>
}
