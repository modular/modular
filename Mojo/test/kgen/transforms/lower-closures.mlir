// RUN: kgen-opt -lower-closures -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.func @execute_byref_async_closure_0
// CHECK-SAME:    (%arg0: index, %arg1: !kgen.pointer<index> byref_result) async {
// CHECK-NEXT:    store %arg0, %arg1

// CHECK-LABEL: kgen.func @execute_byref_async_closure_1
// CHECK-SAME:    (%arg0: index, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async {
// CHECK-NEXT:    store %arg0, %arg1
// CHECK-NEXT:    store %arg0, %arg2

// CHECK-LABEL: kgen.func @execute_byref
kgen.func @execute_byref(%arg0: index) {
  // CHECK-NEXT: co.invoke[{{.*}}: @execute_byref_async_closure_0](%arg0)
  co.execute (%arg1: !kgen.pointer<index> byref_result) {
    pop.store %arg0, %arg1 : !kgen.pointer<index>
    hlcf.return
  }
  // CHECK-NEXT: co.invoke[{{.*}}: @execute_byref_async_closure_1](%arg0)
  co.execute (%arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) {
    pop.store %arg0, %arg1 : !kgen.pointer<index>
    pop.store %arg0, %arg2 : !kgen.pointer<index>
    hlcf.return
  }
  hlcf.return
}

// CHECK-LABEL: kgen.func @async_execute_async_closure
// CHECK-SAME: (%arg0: index) async -> index {
// CHECK-NEXT: hlcf.return %arg0

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) async -> index
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: %0 = index.add %idx1, %arg0
// CHECK-NEXT: %1 = index.add %0, %arg1
// CHECK-NEXT: hlcf.return %1

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) async
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_{{[0-9]}}](%arg0, %arg1)
// CHECK-NEXT: hlcf.return

// CHECK-LABEL: kgen.func @async_execute
kgen.func @async_execute(%arg0: index) {
  // CHECK: index.add
  // CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_0](%arg0)
  // CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_2](%arg0, %0)
  %arg1 = index.add %arg0, %arg0
  %0 = co.execute : index {
    hlcf.return %arg0 : index
  }
  %1 = co.execute {
    %idx1 = index.constant 1
    %2 = co.execute : index {
      %3 = index.add %idx1, %arg0
      %4 = index.add %3, %arg1
      hlcf.return %4 : index
    }
    hlcf.return
  }
  hlcf.return
}

// CHECK-LABEL: kgen.func @co_await
kgen.func @co_await(%arg0: !co.routine, %arg1: !kgen.pointer<index>) {
  // CHECK-NEXT: set_byref_error_result %arg0(%arg1, %arg1)
  // CHECK-NEXT: co.suspend (%hdl)
  // CHECK-NEXT:   [[CALLBACK:%.*]] = co.get_callback_ptr %arg0
  // CHECK-NEXT:   [[FN_PTR:%.*]] = kgen.struct.gep [[CALLBACK]][0]
  // CHECK-NEXT:   [[HDL_PTR:%.*]] = kgen.struct.gep [[CALLBACK]][1]
  // CHECK-NEXT:   [[FN:%.*]] = co.resume %hdl
  // CHECK-NEXT:   store [[FN]], [[FN_PTR]]
  // CHECK-NEXT:   store %hdl, [[HDL_PTR]]
  // CHECK-NEXT:   [[RESUME:%.*]] = co.resume %arg0
  // CHECK-NEXT:   call_indirect [[RESUME]](%arg0)
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: co.destroy %arg0
  // CHECK-NEXT: return
  co.await %arg0, %arg1, %arg1 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> ()
  hlcf.return
}

// CHECK-LABEL: kgen.func @co_raises
kgen.func @co_raises(%arg0: !co.routine, %arg1: !kgen.pointer<index>, %arg2: !kgen.pointer<index>) -> i1 {
  // CHECK-NEXT: set_byref_error_result %arg0(%arg1, %arg2)
  // CHECK: [[RES:%.*]] = co.get_results %arg0
  // CHECK-NEXT: co.destroy %arg0
  // CHECK-NEXT: return [[RES]]
  %0 = co.await %arg0, %arg1, %arg2 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> i1
  hlcf.return %0 : i1
}
