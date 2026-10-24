// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -mlir-disable-threading \
// RUN:   -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Instantiations of one generator are emitted sorted by their printed input
// parameters at any thread count, so "10" sorts before "2".

// CHECK-LABEL: kgen.func @"inst,n=10"
// CHECK: kgen.func @"inst,n=2"
// CHECK: kgen.func @"inst,n=3"
// CHECK: kgen.func export @entry
kgen.generator @inst<n: index>() -> index {
  %0 = kgen.param.constant = <n>
  hlcf.return %0 : index
}

kgen.generator export @entry() {
  %0 = kgen.call @inst<3>() : () -> index
  %1 = kgen.call @inst<10>() : () -> index
  %2 = kgen.call @inst<2>() : () -> index
  hlcf.return
}
