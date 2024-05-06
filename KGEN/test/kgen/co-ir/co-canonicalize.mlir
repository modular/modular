// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @promise_no_use
kgen.func @promise_no_use(%arg0: !co.routine) {
  %0 = co.promise %arg0 : <index>
  // CHECK-NEXT: kgen.return
  kgen.return
}
