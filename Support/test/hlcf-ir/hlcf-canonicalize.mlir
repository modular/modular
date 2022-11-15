// RUN: support-dialect-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @terminators_conditionally_pure
func.func @terminators_conditionally_pure(%arg0: i1) {
  hlcf.loop {
    // CHECK-NOT: {a}
    hlcf.if %arg0 {
      hlcf.yield
    } else {
      hlcf.yield
    } {a}

    // CHECK: {b}
    hlcf.if %arg0 {
      hlcf.return
    } else {
      hlcf.yield
    } {b}

    // CHECK: {c}
    hlcf.if %arg0 {
      hlcf.continue
    } else {
      hlcf.yield
    } {c}

    // CHECK: {d}
    hlcf.if %arg0 {
      hlcf.break
    } else {
      hlcf.yield
    } {d}

    // CHECK-NOT: {e}
    hlcf.loop {
      hlcf.break
    } {e}

    // CHECK: {f}
    hlcf.loop {
      hlcf.continue
    } {f}

    hlcf.break
  }
  return
}
