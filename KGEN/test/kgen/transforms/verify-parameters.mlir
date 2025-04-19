// RUN: kgen-opt %s -split-input-file -verify-parameters=simplify=true | FileCheck %s

// CHECK-LABEL: no_constrains_deduplication
kgen.generator @no_constrains_deduplication() {
  kgen.param.declare cond = <1>
  kgen.param.if <eq(cond, 1)> {
    kgen.param.declare B0 : !kgen.string = <"foo">
    // CHECK: kgen.param.assert <0>, "foo"
    kgen.param.assert <eq(2, 3)>, B0
    kgen.return
  } else {
    kgen.param.declare B1 : !kgen.string = <"bar">
    // CHECK: kgen.param.assert <0>, "bar"
    kgen.param.assert <eq(2, 3)>, B1
    kgen.param.yield
  }
  kgen.param.declare B2 : !kgen.string = <"baz">
  // CHECK: kgen.param.assert <0>, "baz"
  kgen.param.assert <eq(2, 3)>, B2
  kgen.return
}
