// RUN: kgen-opt -lower-calling-conventions -allow-unregistered-dialect %s | FileCheck %s

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
