// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

// CHECK-LABEL: @nested_ops
kgen.func @nested_ops(%cond: i1) -> index {
  // CHECK: scf.if
  %2 = scf.if %cond -> index {
    // CHECK-NOT: kgen.param.constant
    %0 = kgen.param.constant = <4>
    scf.yield %0 : index
  } else {
    // CHECK-NOT: kgen.param.constant
    %1 = kgen.param.constant = <1>
    scf.yield %1 : index
  }
  kgen.return %2 : index
}
