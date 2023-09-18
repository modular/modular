// RUN: kgen-opt -lower-closures -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.func @coroutine
// CHECK-SAME: (%arg0: i1) -> !pop.coroutine<() -> index> no_inline
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
  %result = lit.async.call[(i1) async -> index: @coroutine](%true)
  kgen.return
}

// CHECK-LABEL: kgen.func @async_execute_async_closure
// CHECK-SAME: (%arg0: index) -> !pop.coroutine<() -> index> no_inline
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> index>
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) -> !pop.coroutine<() -> index> no_inline
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> index>
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: %1 = index.add %idx1, %arg0
// CHECK-NEXT: %2 = index.add %1, %arg1
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) -> !pop.coroutine<() -> ()>
// CHECK-NEXT: %0 = pop.coroutine.handle : <() -> ()>
// CHECK: kgen.call @async_execute_async_closure_{{[0-9]}}(%arg0, %arg1)
// CHECK: kgen.return %0

// CHECK-LABEL: kgen.func @async_execute
kgen.func @async_execute(%arg0: index) {
  // CHECK: index.add
  // CHECK-NEXT: kgen.call @async_execute_async_closure(%arg0)
  // CHECK-NEXT: kgen.call @async_execute_async_closure_{{[0-9]}}(%arg0, %0)
  %arg1 = index.add %arg0, %arg0
  %0 = lit.async.execute <() -> index> {
    lit.async.return %arg0 : index
  }
  %1 = lit.async.execute <() -> ()> {
    %idx1 = index.constant 1
    %2 = lit.async.execute <() -> index> {
      %3 = index.add %idx1, %arg0
      %4 = index.add %3, %arg1
      lit.async.return %4 : index
    }
    lit.async.return
  }
  kgen.return
}

// CHECK: kgen.func @some_closure(%arg0: index) capturing -> index no_inline

// CHECK-LABEL: kgen.func @main_closure_arg
kgen.func @main_closure_arg(%arg0: index) {
  // CHECK: kgen.create_closure [(index) capturing -> index: @some_closure](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  } { name = "some_closure" }
  kgen.return
}

// CHECK: kgen.func @two_captures(%arg0: si32, %arg1: si64, %arg2: index) capturing -> index

// CHECK-LABEL: kgen.func @capturing_region
kgen.func @capturing_region(%arg0: si32, %arg1: si64) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [(si32, si64, index) capturing -> index: @two_captures](%arg0, %arg1)
  %0 = kgen.stage_closure = (%arg2: index) capturing -> index {
    "unregistered_op_to_capture"(%arg0, %arg1) : (si32, si64) -> ()
    kgen.return %arg2 : index
  } { name = "two_captures" }
  %1 = kgen.call_signature %0(%idx4) : (index) capturing -> index
  kgen.return
}

// CHECK: kgen.func @no_name_attr_closure(%arg0: index) capturing -> index
// CHECK: kgen.func @no_name_attr_closure_{{[0-9]}}(%arg0: index) capturing -> index

// CHECK-LABEL: kgen.func @no_name_attr(
kgen.func @no_name_attr(%arg0: index, %arg1: index) {
  // CHECK: kgen.create_closure [(index) capturing -> index: @no_name_attr_closure](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  }
  // CHECK: kgen.create_closure [(index) capturing -> index: @no_name_attr_closure_{{[0-9]}}](%arg1)
  %1 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg1 : index
  } { name = 6 }
  kgen.return
}

// CHECK: kgen.func @constant_in_closure() capturing -> index

// CHECK-LABEL: kgen.func @constant_in(
kgen.func @constant_in(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [() capturing -> index: @constant_in_closure]()
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %idx4 : index
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @create_closure(
kgen.func @create_closure() {
  // CHECK: %0 = kgen.create_closure [() -> (): @create_closure_closure]()
  %0 = kgen.stage_closure = () {
    kgen.return
  }
  kgen.return
}
