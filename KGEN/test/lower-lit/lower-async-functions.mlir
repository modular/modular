// RUN: kgen-opt -lower-async-functions %s | FileCheck %s

// CHECK-LABEL: kgen.func @coroutine
// CHECK-SAME: (%arg0: i1) -> !pop.coroutine<() -> index>
kgen.func @coroutine(%arg0: i1) async -> index {
  // CHECK: %[[HDL:.*]] = pop.coroutine.handle : <() -> index>
  // CHECK: hlcf.if
  hlcf.if %arg0 {
    %idx1 = index.constant 1
    // CHECK: %[[PROMISE:.*]] = pop.coroutine.promise %[[HDL]]
    // CHECK-NEXT: %[[RES:.*]] = pop.struct.gep %[[PROMISE]][0]
    // CHECK-NEXT: pop.store %idx1, %[[RES]]
    // CHECK-NEXT: kgen.return %[[HDL]]
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %idx0 = index.constant 0
  // CHECK: %[[PROMISE:.*]] = pop.coroutine.promise %[[HDL]]
  // CHECK-NEXT: %[[RES:.*]] = pop.struct.gep %[[PROMISE]][0]
  // CHECK-NEXT: pop.store %idx0, %[[RES]]
  // CHECK-NEXT: kgen.return %[[HDL]]
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @call_coroutine
kgen.func @call_coroutine() {
  %true = index.bool.constant true
  // CHECK: kgen.call @coroutine(%true) : (i1) -> !pop.coroutine<() -> index>
  %result = lit.async.call[<>(i1) async -> index: @coroutine](%true)
  kgen.return
}

// CHECK-LABEL: kgen.func @async_execute_async_closure
// CHECK-SAME: (%arg0: index) -> !pop.coroutine<() -> index>
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> index>
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute_async_closure_0
// CHECK-SAME: (%arg0: index, %arg1: index) -> !pop.coroutine<() -> index>
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> index>
// CHECK: index.add
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute_async_closure_1
// CHECK-SAME: (%arg0: index) -> !pop.coroutine<() -> ()>
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> ()>
// CHECK: kgen.call @async_execute_async_closure_0(%idx1, %arg0)
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute
kgen.func @async_execute(%arg0: index) {
  // CHECK-NEXT: kgen.call @async_execute_async_closure(%arg0)
  // CHECK-NEXT: kgen.call @async_execute_async_closure_1(%arg0)
  %0 = lit.async.execute <() -> index> {
    lit.async.return %arg0 : index
  }
  %1 = lit.async.execute <() -> ()> {
    %idx1 = index.constant 1
    %2 = lit.async.execute <() -> index> {
      %3 = index.add %idx1, %arg0
      lit.async.return %3 : index
    }
    lit.async.return
  }
  kgen.return
}
