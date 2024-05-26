// RUN: kgen-opt %s -automatic-inline='func-pipeline=canonicalizer' | FileCheck %s

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  // CHECK-NEXT: return
  kgen.call @callee() :()->()
  kgen.return
}

// CHECK-NOT: kgen.func @callee
kgen.func @callee() {
  %unused = kgen.param.constant = <1>
  kgen.return
}

// CHECK-LABEL: kgen.func @no_callers
kgen.func @no_callers() {
  // CHECK-NEXT: return
  %unused = kgen.param.constant = <1>
  kgen.return
}
