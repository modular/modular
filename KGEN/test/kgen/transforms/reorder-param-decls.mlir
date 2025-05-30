// RUN: kgen-opt %s -split-input-file -reorder-param-decls -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @simple
kgen.generator @simple() {
  // COM: reorder param decl to before use.
  // CHECK: kgen.param.declare q
  // CHECK-NEXT: kgen.param.declare w
  // CHECK-NEXT: kgen.call @g2
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.param.declare q = <3>
  kgen.param.declare w = <5>

  kgen.return
}

// CHECK-LABEL: @nestedRegions()
kgen.generator @nestedRegions() {
  // COM: reorder param decl to before use.
  // CHECK: kgen.param.declare cond_var
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    // CHECK: kgen.param.declare next_lt
    // CHECK-NEXT: "should.not.appear"
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    // CHECK: kgen.param.declare next_gt
    // CHECK-NEXT: "should.appear"
    %3 = "should.appear"() : () -> index
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield %3 : index
  }

  kgen.param.declare cond_var = <32>
  kgen.return
}
