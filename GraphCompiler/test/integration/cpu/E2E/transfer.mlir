// RUN: mo-opt %s --mogg-to-mgp | FileCheck %s

//
mo.graph @transfer_d2d(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %c0 = mo.chain.create()
  %0, %c1 = mo.transfer[%c0] %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mgp.model
//  CHECK:   mgp.buffer.device_to_device
// -----

mo.graph @transfer_d2h(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32> {
  %c0 = mo.chain.create()
  %0, %c1 = mo.transfer[%c0] %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %0 : !mo.tensor<[4, 4], f32>
}
// CHECK-LABEL: mgp.model
//  CHECK:   mgp.buffer.device_to_host
// -----

mo.graph @transfer_h2d(%arg0: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %c0 = mo.chain.create()
  %0, %c1 = mo.transfer[%c0] %arg0 : !mo.tensor<[4, 4], f32> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mgp.model
//  CHECK:   mgp.buffer.host_to_device
