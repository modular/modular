// RUN: graphrt-opt %s -canonicalize -allow-unregistered-dialect -split-input-file | FileCheck %s --enable-var-scope

// Eliminate chain operands whose readiness is implied by other operands.

// CHECK-LABEL: mef.callable @eliminate_implied_chains
// CHECK: (%[[CH0:.*]]: !mef.chain) -> !mef.chain
mef.callable @eliminate_implied_chains(%ch0: !mef.chain) -> !mef.chain {

  // CHECK: %[[OUT:.*]] = "foo"(%[[CH0]]) : (!mef.chain) -> !mef.chain
  %ch1 = "foo"(%ch0) : (!mef.chain) -> (!mef.chain)
  %ch2 = grt.chain.create(%ch0, %ch1)

  // mef.output %[[OUT]] : !mef.chain
  mef.output %ch2 : !mef.chain
}

// -----

// CHECK-LABEL: mef.callable @eliminate_implied_chains_control_flow
// CHECK: (%[[CH0:.*]]: !mef.chain, %[[COND:.*]]: i1) -> !mef.chain
mef.callable @eliminate_implied_chains_control_flow(%ch0: !mef.chain, %cond: i1) -> !mef.chain {

  // CHECK: %[[IF:.*]] = grt.if %[[COND]]
  %res1 = grt.if %cond (%ch0: !mef.chain) -> !mef.chain {
    mef.output %ch0 : !mef.chain
  } else {
    mef.output %ch0 : !mef.chain
  }

  // The readiness of %ch0 is implied by the readiness of %cond, as grt.if is
  // strict in the condition argument.

  // CHECK-NOT: grt.chain.create
  %ch2 = grt.chain.create(%cond: i1, %res1)

  // mef.output %[[IF]] : !mef.chain
  mef.output %ch2 : !mef.chain
}

// -----

// CHECK-LABEL: mef.callable @eliminate_implied_chains_control_flow
// CHECK: (%[[CH0:.*]]: !mef.chain, %[[COND:.*]]: i1) -> !mef.chain
mef.callable @eliminate_implied_chains_control_flow(%ch0: !mef.chain, %cond: i1) -> !mef.chain {

  // CHECK: %[[IF:.*]] = grt.if %[[COND]]
  %res1 = grt.if %cond (%ch0: !mef.chain) -> !mef.chain {
    mef.output %ch0 : !mef.chain
  } else {
    mef.output %ch0 : !mef.chain
  }

  // The readiness of %ch0 is not implied by the result chain of the grt.if,
  // as grt.if is only strict in its condition argument.

  // CHECK: %[[OUT:.*]] = grt.chain.create (%[[CH0]], %[[IF]])
  %ch2 = grt.chain.create(%ch0, %res1)

  // mef.output %[[OUT]] : !mef.chain
  mef.output %ch2 : !mef.chain
}

