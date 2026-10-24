// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mef.callable @if
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: si32, %[[ARG2:.*]]: si32
mef.callable @if(%cond: i1, %v1: si32, %v2: si32) -> (si32, si32) {
  // CHECK: %[[RES1:.*]] = grt.if %[[ARG0]] (%[[ARG1]]: si32, %[[ARG2]]: si32) -> si32 {
  %res1 = grt.if %cond (%v1: si32, %v2: si32) -> si32 {
    // CHECK-NEXT: mef.output %[[ARG1]]
    mef.output %v1 : si32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output %[[ARG2]]
    mef.output %v2 : si32
  }

  // CHECK: %[[RES2:.*]] = grt.if %[[ARG0]] (%[[ARG1]] as %[[INNER1:.*]]: si32, %[[ARG1]] as %[[INNER2:.*]]: si32) -> si32 {
  %res2 = grt.if %cond (%v1: si32, %v1 as %x2: si32) -> si32 {
    // CHECK-NEXT: mef.output %[[INNER1]]
    mef.output %v1 : si32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output %[[INNER2]]
    mef.output %x2 : si32
  }

  // CHECK: grt.if %[[ARG0]] () -> () {
  grt.if %cond () -> () {
    // CHECK-NEXT: mef.output
    mef.output
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output
    mef.output
  // CHECK-NEXT: }
  }

  // CHECK: mef.output %[[RES1]], %[[RES2]]
  mef.output %res1, %res2 : si32, si32
}

// CHECK-LABEL: mef.callable @while
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32
mef.callable @while(%v1: i32, %v2: f32) -> (i32, f32, i32, i32) {
  // CHECK: %[[RESA:.*]]:2 = grt.while (%[[ARG1]]: i32, %[[ARG2]]: f32) {
  %res1, %res2 = grt.while (%v1: i32, %v2: f32) {
    // CHECK-NEXT: %[[COND:.*]] = "my.cond"(%[[ARG1]], %[[ARG2]])
    %c = "my.cond"(%v1, %v2) : (i32, f32) -> i1
    // CHECK-NEXT: mef.output %[[ARG1]], %[[ARG2]], %[[COND]]
    mef.output %v1, %v2, %c : i32, f32, i1
  // CHECK-NEXT: } do {
  } do {
    // CHECK-NEXT: mef.output %[[ARG1]], %[[ARG2]]
    mef.output %v1, %v2 : i32, f32
  // CHECK-NEXT: }
  }

  // CHECK: %[[RESB:.*]]:2 = grt.while (%[[ARG1]] as %[[INNER1:.*]]: i32, %[[ARG1]] as %[[INNER2:.*]]: i32) {
  %res3, %res4 = grt.while (%v1: i32, %v1 as %x: i32) {
    // CHECK-NEXT: %[[COND:.*]] = "my.cond"(%[[INNER1]], %[[INNER2]])
    %c = "my.cond"(%v1, %x) : (i32, i32) -> i1
    // CHECK-NEXT: mef.output %[[INNER1]], %[[INNER2]], %[[COND]]
    mef.output %v1, %x, %c : i32, i32, i1
  // CHECK-NEXT: } do {
  } do {
    // CHECK-NEXT: mef.output %[[INNER1]], %[[INNER2]]
    mef.output %v1, %x : i32, i32
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: mef.output %[[RESA]]#0, %[[RESA]]#1, %[[RESB]]#0, %[[RESB]]#1
  mef.output %res1, %res2, %res3, %res4 : i32, f32, i32, i32
}

// CHECK-LABEL: mef.callable @repeat
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: f32
mef.callable @repeat(%arg0: si32, %arg1: f32) -> (si32, f32, f32, f32) {
  // CHECK: %[[RESA:.*]]:2 = grt.repeat.i32 %[[ARG0]] (%[[ARG0]]: si32, %[[ARG1]]: f32) {
  %res1, %res2 = grt.repeat.i32 %arg0 (%arg0: si32, %arg1: f32) {
    // CHECK-NEXT: "my.use"(%[[ARG0]], %[[ARG1]])
    "my.use"(%arg0, %arg1) : (si32, f32) -> ()

    mef.output %arg0, %arg1 : si32, f32
  }

  // CHECK: %[[RESB:.*]]:2 = grt.repeat.i32 %[[ARG0]] (%[[ARG1]] as %[[INNER1:.*]]: f32, %[[ARG1]] as %[[INNER2:.*]]: f32) {
  %res3, %res4 = grt.repeat.i32 %arg0 (%arg1: f32, %arg1 as %x: f32) {
    // CHECK-NEXT: "my.use"(%[[INNER1]], %[[INNER2]])
    "my.use"(%arg1, %x) : (f32, f32) -> ()

    mef.output %arg1, %x : f32, f32
  }


  // Zero argument loop.

  // CHECK: grt.repeat.i32 %[[ARG0]] () {
  grt.repeat.i32 %arg0 () {
    mef.output
  }

  // CHECK: mef.output %[[RESA]]#0, %[[RESA]]#1, %[[RESB]]#0, %[[RESB]]#1
  mef.output %res1, %res2, %res3, %res4 : si32, f32, f32, f32
}

// CHECK-LABEL: mef.callable @merge_non_chain_inputs(
mef.callable @merge_non_chain_inputs(
  // CHECK-SAME: %[[ARG0:.*]]: index
  %arg0: index
// CHECK-SAME: ) -> !mef.chain {
) -> !mef.chain {
  // CHECK-NEXT: %[[CH0:.*]] = grt.chain.create ()
  %ch0 = grt.chain.create ()
  // CHECK-NEXT: %[[CH1:.*]] = grt.chain.create ()
  %ch1 = grt.chain.create ()
  // CHECK-NEXT: %[[CH2:.*]] = grt.chain.create (%[[ARG0]]: index, %[[CH1]])
  %ch2 = grt.chain.create (%arg0: index, %ch1)
  // CHECK-NEXT: mef.output %[[CH2]] : !mef.chain
  mef.output %ch2 : !mef.chain
// CHECK-NEXT: }
}
