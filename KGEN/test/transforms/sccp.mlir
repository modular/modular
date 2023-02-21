// RUN: kgen-opt -sccp %s | FileCheck %s

// COM: Check that KGEN ops correctly implement interfaces to interact with
// dataflow analyses.

kgen.generator.interface @iface_is_extern() -> index

// CHECK-LABEL: @branch_iface
kgen.generator @branch_iface(%cond: i1) -> index {
  // CHECK: %[[IF_RESULT:.*]] = hlcf.if
  %0 = hlcf.if %cond -> index {
    %1 = kgen.call @iface_is_extern() : () -> index
    hlcf.yield %1 : index
  } else {
    %1 = index.constant 0
    hlcf.yield %1 : index
  }
  // CHECK: return %[[IF_RESULT]]
  kgen.return %0 : index
}
