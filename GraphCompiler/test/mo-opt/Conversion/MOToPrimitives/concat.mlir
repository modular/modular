// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s

// CHECK-LABEL: mgp.model @concat_dynamic_output
// CHECK:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,

// CHECK: %[[IN_BUF:.*]] = mgp.buffer.alloc
// CHECK: %[[OUT_CHAIN0:.*]] = mogg.routine.async
// CHECK: %[[IN_BUF2:.*]] = mgp.buffer.alloc
// CHECK: %[[OUT_CHAIN1:.*]] = mogg.routine.async
// CHECK: %[[OUT_BUF:.*]] = mgp.buffer.alloc

// CHECK: %[[OUT_CHAIN3:.*]] = grt.chain.create (%[[OUT_CHAIN0]], %[[OUT_CHAIN1]], %[[CH0]]
// CHECK: %[[CH2:.*]] = mgp.buffer.concat(%[[OUT_CHAIN3]], %[[OUT_BUF]], %[[IN_BUF]], %[[IN_BUF2]]

// CHECK: %[[CH3:.*]] = grt.chain.create (%[[CH2]], %[[CH0]])

// COM: Check user of the concat has the correct chain.
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH3]]

mo.graph @concat_dynamic_output<D6, D7>(%arg0: !mo.tensor<[D6], f32>) -> !mo.tensor<[D7], f32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[D6], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[D6], f32>
  %2 = mogg.concat(%0, %1) : (!mo.tensor<[D6], f32>, !mo.tensor<[D6], f32>) -> !mo.tensor<[D7], f32>
  %3 = mo.relu(%2) : !mo.tensor<[D7], f32>
  mo.output %3 : !mo.tensor<[D7], f32>
}
