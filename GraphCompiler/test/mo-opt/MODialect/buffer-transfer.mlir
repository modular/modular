// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK-LABEL: mo.graph @buffer_transfer(
// CHECK-SAME: [[ARG0:%.*]]: !mo.buffer<[2, 2], f32>)
// CHECK-SAME: -> !mo.buffer<[2, 2], f32, gpu:0>
// CHECK: [[CH:%.*]] = mo.chain.create()
// CHECK: [[DARG0:%.*]] = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>
// CHECK: mo.buffer.transfer([[ARG0]], [[DARG0]], [[CH]]) : !mo.buffer<[2, 2], f32>, !mo.buffer<[2, 2], f32, gpu:0>
// CHECK: mo.output [[DARG0]] : !mo.buffer<[2, 2], f32, gpu:0>
mo.graph @buffer_transfer(%arg0: !mo.buffer<[2, 2], f32>) -> !mo.buffer<[2, 2], f32, gpu:0>  {
  %ch = mo.chain.create()
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>

  mo.buffer.transfer(%arg0, %darg0, %ch) : !mo.buffer<[2, 2], f32>, !mo.buffer<[2, 2], f32, gpu:0>

  mo.output %darg0: !mo.buffer<[2, 2], f32, gpu:0>
}
