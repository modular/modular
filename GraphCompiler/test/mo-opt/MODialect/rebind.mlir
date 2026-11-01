// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: mo.graph @rebind_w_params(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[2, 3], si8> {
mo.graph @rebind_w_params(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[2, 3], si8> {
  // CHECK: %[[VAL0:.*]] = mo.rebind<() -> [[N:.*]], [[M:.*]]>(%[[ARG0]]) :
  // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[[N]], [[M]]], f32>
  %0 = mo.rebind<() -> N, M>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  // CHECK: %[[VAL1:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[[[N]], [[M]]], f32> -> !mo.tensor<[[[N]], [[M]]], f16>
  %1 = mo.rebind(%0) : !mo.tensor<[N, M], f32> -> !mo.tensor<[N, M], f16>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[[[N]], [[M]]], f16> -> !mo.tensor<[2, 3], si8>
  %2 = mo.rebind(%1) : !mo.tensor<[N, M], f16> -> !mo.tensor<[2, 3], si8>
  // CHECK-NEXT: mo.output %[[VAL2]] : !mo.tensor<[2, 3], si8>
  mo.output %2 : !mo.tensor<[2, 3], si8>
}

// CHECK: mo.graph @rebind_in_if(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) {
mo.graph @rebind_in_if(%arg0: !mo.tensor<[?, ?], f32>, %cond : !mo.tensor<[], bool>) {
  // CHECK-NEXT: %[[VAL5:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[?, ?], f32> {
  %5 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[?, ?], f32> {
    // CHECK-NEXT: %[[VAL3:.*]] = mo.rebind<() -> [[N2:.*]], [[M2:.*]]>(%[[ARG0]]) :
    // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[[N2]], [[M2]]], f32>
    %3 = mo.rebind<() -> N2, M2>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N2, M2], f32>
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[?, ?], f32>
    mo.yield %arg0 : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: %[[VAL4:.*]] = mo.rebind<() -> [[N3:.*]], [[M3:.*]]>(%[[ARG0]]) :
    // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[[N3]], [[M3]]], f32>
    %4 = mo.rebind<() -> N3, M3>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N3, M3], f32>
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[?, ?], f32>
    mo.yield %arg0 : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: mo.output
  mo.output
}
