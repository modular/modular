// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @terminators_conditionally_pure
func.func @terminators_conditionally_pure(%arg0: i1) {
  hlcf.loop {
    // CHECK: {b}
    hlcf.if %arg0 {
      kgen.return
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

    // CHECK: {e}
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

// CHECK-LABEL: @fold_if_return
kgen.func @fold_if_return(%arg0 : index, %arg1: index, %arg2: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK-NEXT: kgen.return %arg0
  // CHECK-NOT: kgen.return
  %cond = kgen.param.constant: i1 = <1>
  hlcf.if %cond {
    kgen.return %arg0: index
  } else {
    kgen.return %arg1: index
  }
  kgen.return %arg2: index
}

// CHECK-LABEL: @fold_if_yield
kgen.func @fold_if_yield(%arg0 : index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK-NEXT: %[[TEN:.*]] = index.constant 10
  // CHECK-NEXT: %[[RES:.*]] = index.add %arg1, %[[TEN]]
  // CHECK-NEXT: kgen.return %[[RES]]
  %cond = kgen.param.constant: i1 = <0>
  %z = hlcf.if %cond -> index {
    hlcf.yield %arg0: index
  } else {
    hlcf.yield %arg1: index
  }
  %ten = index.constant 10
  %r = index.add %z, %ten
  kgen.return %r: index
}
