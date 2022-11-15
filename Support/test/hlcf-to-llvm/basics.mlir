// RUN: support-dialect-opt -lower-hlcf-to-llvm -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @nested_continue
func.func @nested_continue(%arg0: i1) {
  // CHECK-NEXT: llvm.br ^bb1
  hlcf.loop {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llvm.cond_br %arg0, ^bb2, ^bb3
    hlcf.if %arg0 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.return
      hlcf.return
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: llvm.br ^bb1
      hlcf.continue
    }
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: llvm.br ^bb5
    hlcf.break
  }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: @nested_break
func.func @nested_break(%arg0: i1) {
  // CHECK-NEXT: llvm.br ^bb1
  hlcf.loop {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llvm.cond_br %arg0, ^bb2, ^bb3
    hlcf.if %arg0 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.br ^bb4
      hlcf.yield
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: llvm.br ^bb5
      hlcf.break
    }
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: llvm.br ^bb1
    hlcf.continue
  }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: @deeply_nested
func.func @deeply_nested(%arg0: i1, %arg1: i1, %arg2: i1) {
  // CHECK-NEXT: llvm.cond_br %arg0, ^bb1, ^bb10
  hlcf.if %arg0 {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llvm.br ^bb2
    hlcf.loop {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.cond_br %arg1, ^bb3, ^bb7
      hlcf.if %arg1 {
        // CHECK-NEXT: ^bb3:
        // CHECK-NEXT: llvm.cond_br %arg2, ^bb4, ^bb5
        hlcf.if %arg2 {
          // CHECK-NEXT: ^bb4:
          // CHECK-NEXT: llvm.br ^bb9
          hlcf.break
        } else {
          // CHECK-NEXT: ^bb5:
          // CHECK-NEXT: llvm.br ^bb6
          hlcf.yield
        }
        // CHECK-NEXT: ^bb6:
        // CHECK-NEXT: llvm.br ^bb2
        hlcf.continue
      } else {
        // CHECK-NEXT: ^bb7:
        // CHECK-NEXT: llvm.br ^bb8
        hlcf.yield
      }
      // CHECK-NEXT: ^bb8:
      // CHECK-NEXT: llvm.br ^bb2
      hlcf.continue
    }
    // CHECK-NEXT: ^bb9:
    // CHECK-NEXT: llvm.return
    hlcf.return
  } else {
    // CHECK-NEXT: ^bb10:
    // CHECK-NEXT: llvm.br ^bb11
    hlcf.yield
  }
  // CHECK-NEXT: ^bb11:
  return
}

// CHECK-LABEL: @two_trees
func.func @two_trees(%arg0: i1) {
  // CHECK-NEXT: llvm.br ^bb1
  hlcf.loop {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llvm.cond_br %arg0, ^bb2, ^bb3
    hlcf.if %arg0 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.br ^bb5
      hlcf.break
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: foo.region
      "foo.region"() ({
        // CHECK-NEXT: llvm.br ^bb1
        hlcf.loop {
          // CHECK-NEXT: ^bb1:
          // CHECK-NEXT: llvm.cond_br %arg0, ^bb2, ^bb3
          hlcf.if %arg0 {
            // CHECK-NEXT: ^bb2:
            // CHECK-NEXT: llvm.br ^bb4
            hlcf.yield
          } else {
            // CHECK-NEXT: ^bb3:
            // CHECK-NEXT: llvm.br ^bb1
            hlcf.continue
          }
          // CHECK-NEXT: ^bb4:
          // CHECK-NEXT: llvm.br ^bb5
          hlcf.break
        }
        // CHECK-NEXT: ^bb5:
        // CHECK-NEXT: foo.terminator
        "foo.terminator"() : () -> ()
      // CHECK-NEXT: }) : () -> ()
      }) : () -> ()
      // CHECK-NEXT: llvm.br ^bb4
      hlcf.yield
    }
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: llvm.br ^bb1
    hlcf.continue
  }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: @operands_and_results
func.func @operands_and_results(%arg0: i1, %arg1: i32, %arg2: i64) -> (i32, i64) {
  // CHECK-NEXT: %[[INIT0:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-NEXT: %[[INIT1:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-NEXT: llvm.br ^bb1(%[[INIT0]], %[[INIT1]] : i1, i32)
  %0:2 = hlcf.loop (%0 = %arg0 : i1, %1 = %arg1 : i32) -> (i32, i64) {
    // CHECK-NEXT: ^bb1(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32):
    // CHECK-NEXT: %[[V0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
    // CHECK-NEXT: %[[V1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
    // CHECK-NEXT: llvm.cond_br %[[V0]], ^bb2, ^bb3
    %2 = hlcf.if %0 -> i32 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %arg2
      // CHECK-NEXT: llvm.br ^bb8(%[[R0]], %[[R1]] : i32, i64
      hlcf.break %arg1, %arg2 : i32, i64
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: llvm.br ^bb4(%[[R0]] : i32)
      hlcf.yield %arg1 : i32
    }
    // CHECK-NEXT: ^bb4(%[[ARG2:.*]]: i32):
    // CHECK-NEXT: %[[V2:.*]] = builtin.unrealized_conversion_cast %[[ARG2]]
    // CHECK-NEXT: llvm.cond_br %[[V0]], ^bb5, ^bb6
    %3 = hlcf.if %0 -> i1 {
      // CHECK-NEXT: ^bb5:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg0
      // CHECK-NEXT: llvm.br ^bb7(%[[R0]] : i1)
      hlcf.yield %arg0 : i1
    } else {
      // CHECK-NEXT: ^bb6:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg0
      // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: llvm.br ^bb1(%[[R0]], %[[R1]] : i1, i32
      hlcf.continue %arg0, %arg1 : i1, i32
    }
    // CHECK-NEXT: ^bb7(%[[ARG3:.*]]: i1):
    // CHECK-NEXT: %[[V3:.*]] = builtin.unrealized_conversion_cast %[[ARG3]]
    // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[V3]]
    // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %[[V2]]
    // CHECK-NEXT: llvm.br ^bb1(%[[R0]], %[[R1]] : i1, i32)
    hlcf.continue %3, %2 : i1, i32
  }
  // CHECK-NEXT: ^bb8(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i64):
  // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK-NEXT: return %[[R0]], %[[R1]]
  return %0#0, %0#1 : i32, i64
}

// CHECK-LABEL: @multiple_return
func.func @multiple_return(%arg0: i1) -> (i1, i1) {
  hlcf.if %arg0 {
    // CHECK: ^bb1:
    // CHECK-NEXT: %[[V0:.*]] = builtin.unrealized_conversion_cast %arg0
    // CHECK-NEXT: %[[V1:.*]] = builtin.unrealized_conversion_cast %arg0
    // CHECK-NEXT: %[[S0:.*]] = llvm.mlir.undef
    // CHECK-NEXT: %[[S1:.*]] = llvm.insertvalue %[[V0]], %[[S0]][0]
    // CHECK-NEXT: %[[S2:.*]] = llvm.insertvalue %[[V1]], %[[S1]][1]
    // CHECK-NEXT: llvm.return %[[S2]]
    hlcf.return %arg0, %arg0 : i1, i1
  } else {
    hlcf.yield
  }
  return %arg0, %arg0 : i1, i1
}
