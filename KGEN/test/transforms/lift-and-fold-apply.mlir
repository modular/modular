// RUN: kgen-opt -verify-parameters -lift-and-fold-apply %s | FileCheck %s

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

kgen.generator @take_and_pass<N>() -> !pop.array<apply(:(index) -> index @pass, N), index> {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  kgen.param.declare p0 = <1>
  kgen.param.declare p1 = <add(p0, 1)>
  // CHECK: apply *[[L0:.*]] = [(index) -> index: @pass](p0)
  // CHECK: apply *[[L1:.*]] = [(index) -> index: @pass](*[[L0]])
  // CHECK: constant = <*[[L1]]>
  kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
  // CHECK: constant = <*[[L1]]>
  kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
  // CHECK: constant = <*[[L0]]>
  kgen.param.constant = <apply(:(index) -> index @pass, p0)>
  // CHECK: apply *[[L2:.*]] = [(index) -> index: @pass](p1)
  // CHECK: apply *[[L3:.*]] = [(index) -> index: @pass](*[[L2]])
  // CHECK: call @take_and_pass<*[[L2]]>() : () -> !pop.array<*[[L3]], index>
  %0 = kgen.call @take_and_pass<apply(:(index) -> index @pass, p1)>() : ()
    -> !pop.array<apply(:(index) -> index @pass, apply(:(index) -> index @pass, p1)), index>

  // CHECK: region F = <p0>
  kgen.param.declare.region F = <p0>() {
    // CHECK: apply *[[L4:.*]] = [(index) -> index: @pass](p0)
    // CHECK: constant = <*[[L4]]>
    kgen.param.constant = <apply(:(index) -> index @pass, p0)>
    kgen.return
  }

  kgen.return
}
