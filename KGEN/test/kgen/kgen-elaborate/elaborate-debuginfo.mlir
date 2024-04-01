// RUN: kgen-opt %s -split-input-file -elaborate-generators="elaborate-debuginfo=true allow-multiple-primary-impls=true" -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: kgen.func @loc_ref
kgen.generator @loc_ref() {
  kgen.param.if <0> {
    kgen.param.yield
  } else {
    kgen.param.declare A = <1>
    // CHECK: constant = <1> loc([[LOC:#.*]])
    kgen.param.constant = <1> loc(fused<#kgen.param.decl.ref<"A">:index>["a":0:0])
    kgen.param.yield
  }
  kgen.return
}

// CHECK: [[LOC]] = loc(fused<1 : index>
