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
  %result = lit.async_call[<>(i1) async -> index: @coroutine](%true)
  kgen.return
}
