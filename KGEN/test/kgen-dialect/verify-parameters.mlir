// RUN: kgen-opt %s -verify-parameters | FileCheck %s

// CHECK-LABEL: kgen.generator @parameterIsolatedRegions
kgen.generator @parameterIsolatedRegions<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region Fn = <B>() {
    kgen.param.constant = <B>
    kgen.return
  }
  // CHECK: {isolated}

  // CHECK: kgen.param.if
  kgen.param.if <lt(A, 1)> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK: {elseIsolated, thenIsolated}
  kgen.return
}
