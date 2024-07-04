// RUN: kgen-opt %s --lower-custom-ops | FileCheck %s

module {
  kgen.custom.op_impls @__CustomOpImplSymbol [<"custom.arith.neg", :!kgen.signature<(!pop.scalar<si32>) -> !pop.scalar<si32>> @my_impl>]

  kgen.generator @my_impl(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    kgen.return %arg0 : !pop.scalar<si32>
  }
  // CHECK-LABEL: kgen.generator @main(%{{.*}}: !pop.scalar<si32>) -> !pop.scalar<si32> {
  // CHECK-NEXT:    %{{.*}} = kgen.call @my_impl(%{{.*}}) : (!pop.scalar<si32>) -> !pop.scalar<si32>
  // CHECK-NEXT:    kgen.return %{{.*}} : !pop.scalar<si32>
  // CHECK-NEXT:  }
  kgen.generator @main(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    %res = "custom.arith.neg"(%arg0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    kgen.return %res : !pop.scalar<si32>
  }
}
