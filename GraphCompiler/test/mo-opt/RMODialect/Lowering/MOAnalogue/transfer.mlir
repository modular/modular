// This file tests the proper handling of explicit devices on tensors for RMO-MO Analogue ops
// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

//
mo.graph @transfer_d2d(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %ch0 = mo.chain.create()
  %0, %ch1 = rmo.mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_d2d([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL:[A-Za-z0-9]*]]:[[IN_DEV_ID:[0-9]*]]>) -> !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL:[A-Za-z0-9]*]]:[[OUT_DEV_ID:[0-9]*]]> {
//  CHECK-NEXT: [[CHAIN:%.*]] = mo.chain.create
//  CHECK-NEXT: [[V1:%.*]], %{{.*}} = mo.transfer[[[CHAIN]]] [[ARG0]] : !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL]]:[[IN_DEV_ID]]> to <"[[OUT_DEV_LABEL]]", [[OUT_DEV_ID]]>
//  CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL]]:[[OUT_DEV_ID]]>

// -----

mo.graph @transfer_d2h(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32> {
  %ch0 = mo.chain.create()
  %0, %ch1 = rmo.mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %0 : !mo.tensor<[4, 4], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_d2h([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL:[A-Za-z0-9]*]]:[[IN_DEV_ID:[0-9]*]]>) -> !mo.tensor<[4, 4], f32> {
//  CHECK-NEXT: [[CHAIN:%.*]] = mo.chain.create
//  CHECK-NEXT: [[V1:%.*]], %{{.*}} = mo.transfer[[[CHAIN]]] [[ARG0]] : !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL]]:[[IN_DEV_ID]]> to <"cpu", 0>
//  CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[4, 4], f32>

// -----

mo.graph @transfer_h2d(%arg0: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %ch0 = mo.chain.create()
  %0, %ch1 = rmo.mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 4], f32> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_h2d([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL:[A-Za-z0-9]*]]:[[OUT_DEV_ID:[0-9]*]]> {
//  CHECK-NEXT: [[CHAIN:%.*]] = mo.chain.create
//  CHECK-NEXT: [[V1:%.*]], %{{.*}} = mo.transfer[[[CHAIN]]] [[ARG0]] : !mo.tensor<[4, 4], f32> to <"[[OUT_DEV_LABEL]]", [[OUT_DEV_ID]]>
//  CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL]]:[[OUT_DEV_ID]]>
