// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --canonicalize --mogg-to-mgp --mgp-lower-primitives | FileCheck %s

// CHECK-LABEL: @while_op
// CHECK: } execute
// CHECK-NEXT: grt.while
// CHECK: } do {

mo.graph @while_op(%x: !mo.tensor<[], f32>, %arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
   %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.while.condition(%cond) %xi : !mo.tensor<[], f32>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }
   mo.output %res : !mo.tensor<[], f32>
}

// CHECK-LABEL: @multi_while_op
// CHECK: } execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[WHILE:.*]]:5 = grt.while (%[[CH0]] as %[[CH0_IN:.*]]: !mef.chain,
// CHECK: } do {
// CHECK: %[[CH:.*]] = grt.chain.create (%[[CH0_IN]]
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH]]{{.*}} -> !mef.chain {

// CHECK: grt.while (%[[WHILE]]#0 as %[[CH0_IN:.*]]: !mef.chain,{{.*}}, %[[WHILE]]#0 as %[[CH1_IN:.*]]: !mef.chain
// CHECK: } do {
// CHECK: %[[CH0:.*]] = grt.chain.create (%[[CH1_IN]],
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH0]]{{.*}} -> !mef.chain {

mo.graph @multi_while_op(%x: !mo.tensor<[], f32>, %arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
   %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.while.condition(%cond) %xi : !mo.tensor<[], f32>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }

   %res2 = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.while.condition(%cond) %xi : !mo.tensor<[], f32>
   } do {
     %new_x = mo.add (%res, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }

   mo.output %res2 : !mo.tensor<[], f32>
}
