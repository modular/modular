// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK-LABEL: mo.graph @basic
// CHECK: %[[BUFFER:.*]] = mo.buffer.create : !mo.buffer<[2, 2], f32, cuda:0>
// CHECK: mo.output %[[BUFFER]] : !mo.buffer<[2, 2], f32, cuda:0>
mo.graph @basic() -> !mo.buffer<[2, 2], f32, cuda:0>  {
  %buffer = mo.buffer.create : !mo.buffer<[2, 2], f32, cuda:0>
  mo.output %buffer: !mo.buffer<[2, 2], f32, cuda:0>
}

// CHECK-LABEL: mo.graph @basic_parameterized
// CHECK: %[[BUFFER:.*]] = mo.buffer.create : !mo.buffer<[d0, d1], f32, cuda:0>
// CHECK: mo.output %[[BUFFER]] : !mo.buffer<[d0, d1], f32, cuda:0>
mo.graph @basic_parameterized<d0, d1>() -> !mo.buffer<[d0, d1], f32, cuda:0>{
  %buffer = mo.buffer.create : !mo.buffer<[d0, d1], f32, cuda:0>
  mo.output %buffer: !mo.buffer<[d0, d1], f32, cuda:0>
}
