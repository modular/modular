// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @loop
kgen.func @loop(%arg0: i32, %arg1: i64) {
  // CHECK: hlcf.loop {
  hlcf.loop {
    hlcf.break
  }

  // CHECK: hlcf.loop {
  hlcf.loop () -> () {
    hlcf.break
  }

  // CHECK: hlcf.loop (%{{.*}} = %arg0 : i32) {
  hlcf.loop (%0 = %arg0 : i32) -> () {
    hlcf.break
  }

  // CHECK: %{{.*}} = hlcf.loop () -> index {
  %0 = hlcf.loop () -> index {
    hlcf.continue
  }

  // CHECK: %{{.*}}:2 = hlcf.loop () -> (index, index) {
  %1:2 = hlcf.loop () -> (index, index) {
    hlcf.continue
  }

  kgen.return
}

// CHECK-LABEL: kgen.func @if
kgen.func @if(%arg0: i1, %arg1: i32, %arg2: i64) {
  // CHECK-NEXT: hlcf.if %arg0 {
  hlcf.if %arg0 {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: }
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32 {
  %0 = hlcf.if %arg0 -> i32 {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32, i64
  %1:2 = hlcf.if %arg0 -> i32, i64 {
    // CHECK-NEXT: hlcf.yield %arg1, %arg2 : i32, i64
    hlcf.yield %arg1, %arg2 : i32, i64
  } else {
    hlcf.yield %arg1, %arg2 : i32, i64
  }

  kgen.return
}

// CHECK-LABEL: kgen.func @func_loop_if
kgen.func @func_loop_if(%arg0: i1, %arg1: i32, %arg2: i64) -> i32 {
  // CHECK: %[[V0:.*]] = hlcf.loop (%[[A:.*]] = %arg2 : i64) -> i32
  %2 = hlcf.loop (%0 = %arg2 : i64) -> i32 {
    // CHECK: %[[V1:.*]] = hlcf.if %arg0 -> i64
    %1 = hlcf.if %arg0 -> i64 {
      // CHECK: kgen.return %arg1 : i32
      kgen.return %arg1 : i32
    } else {
      // CHECK: hlcf.yield %[[A]] : i64
      hlcf.yield %0 : i64
    }
    // CHECK: hlcf.if %arg0
    hlcf.if %arg0 {
      // CHECK: hlcf.continue %[[A]] : i64
      hlcf.continue %0 : i64
    } else {
      hlcf.yield
    }
    // CHECK: hlcf.break %arg1 : i32
    hlcf.break %arg1 : i32
  }
  kgen.return %2 : i32
}

// CHECK-LABEL: @labelled_loops
kgen.func @labelled_loops(%cond: i1, %arg0: index, %arg1: i32) {
  // CHECK-NEXT: hlcf.loop "foo" (%{{.*}} = %{{.*}} : index) -> i32
  %0 = hlcf.loop "foo" (%a0 = %arg0 : index) -> i32 {
    // CHECK-NEXT: hlcf.loop "bar" (%{{.*}} = %{{.*}} : i32) -> index
    %1 = hlcf.loop "bar" (%a1 = %arg1 : i32) -> index {
      hlcf.if %cond {
        // CHECK: break %{{.*}} : index
        hlcf.break %a0 : index
      } else {
        // CHECK: break "bar" %{{.*}} : index
        hlcf.break "bar" %a0 : index
      }
      hlcf.if %cond {
        // CHECK: break "foo" %{{.*}} : i32
        hlcf.break "foo" %a1 : i32
      } else {
        // CHECK: continue %{{.*}} : i32
        hlcf.continue %a1 : i32
      }
      // CHECK: continue "foo" %{{.*}} : index
      hlcf.continue "foo" %a0 : index
    }
    // CHECK: break %{{.*}} : i32
    hlcf.break %arg1 : i32
  }
  kgen.return
}

// CHECK-LABEL: @switch
kgen.func @switch(%arg0: index, %arg1: i32, %arg2: i64) {
  // CHECK-NEXT: hlcf.switch %arg0
  hlcf.switch %arg0
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: return
    kgen.return
  }
  // CHECK: case 2 {
  case 2 {
    // CHECK-NEXT: yield
    hlcf.yield
  }
  // CHECK: hlcf.switch %arg0 -> i32, i64
  %0:2 = hlcf.switch %arg0 -> i32, i64
  default {
    hlcf.yield %arg1, %arg2 : i32, i64
  }
  kgen.return
}

// CHECK-LABEL: @elif
kgen.func @elif(%arg0: index, %arg1: index, %arg2: index) {
  %idx0 = index.constant 0
  // CHECK:         [[VAR0:%.*]] = hlcf.elif -> index {
  // CHECK-NEXT:      [[VAR1:%.*]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT:      hlcf.elif.yield [[VAR1]]
  // CHECK-NEXT:    } then {
  // CHECK-NEXT:     hlcf.yield %arg0 : index
  // CHECK-NEXT:    } {
  // CHECK-NEXT:     %idx1 = index.constant 1
  // CHECK-NEXT:     [[VAR2:%.*]] = index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:     hlcf.elif.yield [[VAR2]]
  // CHECK-NEXT:    } then {
  // CHECK-NEXT:     hlcf.yield %arg0 : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.yield %arg0 : index
  // CHECK-NEXT:   }
  %0 = hlcf.elif -> index {
    %c = index.cmp eq(%arg0, %idx0)
    hlcf.elif.yield %c
  } then {
    hlcf.yield %arg0 : index
  } {
    %idx1 = index.constant 1
    %c = index.cmp eq(%arg0, %idx1)
    hlcf.elif.yield %c
  } then {
    hlcf.yield %arg0 : index
  } else {
    hlcf.yield %arg0 : index
  }


  // CHECK:      hlcf.elif {
  // CHECK-NEXT:   [[VAR3:%.*]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT:    hlcf.elif.yield [[VAR3]]
  // CHECK-NEXT:  } then {
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  }
  hlcf.elif {
    %c = index.cmp eq(%arg0, %idx0)
    hlcf.elif.yield %c
  } then {
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL:  kgen.func @elifWithArgs
kgen.func @elifWithArgs(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  // CHECK: [[V0:%*.]]:2 = hlcf.elif -> index, index {
  // CHECK-NEXT:   [[V2:%*.]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT:   hlcf.elif.yield [[V2]], %arg0, %arg0 : index, index
  // CHECK-NEXT: } then (%arg1: index, %arg2: index){
  // CHECK-NEXT:   hlcf.yield %arg1, %arg1 : index, index
  // CHECK-NEXT: } (%arg1: index, %arg2: index){
  // CHECK-NEXT:   [[V2:%*.]] = index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:   hlcf.elif.yield [[V2]], %arg1, %arg1 : index, index
  // CHECK-NEXT: } then (%arg1: index, %arg2: index){
  // CHECK-NEXT:   hlcf.yield %arg1, %arg1 : index, index
  // CHECK-NEXT: } else (%arg1: index, %arg2: index){
  // CHECK-NEXT:   hlcf.yield %idx0, %arg1 : index, index
  // CHECK-NEXT: }
  %0:2 = hlcf.elif -> index, index {
     %3 = index.cmp eq(%arg0, %idx0)
     hlcf.elif.yield %3, %arg0, %arg0 : index, index
  } then (%arg1: index, %arg2: index) {
     hlcf.yield %arg1, %arg1 : index, index
  } (%arg1: index, %arg2: index) {
     %4 = index.cmp eq(%arg0, %idx1)
     hlcf.elif.yield %4, %arg1, %arg1 : index, index
  } then (%arg1: index, %arg2: index) {
     hlcf.yield %arg1, %arg1 : index, index
  } else (%arg1: index, %arg2: index) {
     hlcf.yield %idx0, %arg1 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}
