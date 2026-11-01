// RUN: mo-opt --mogg-to-mgp %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: mgp.model @no_params
// CHECK: execute ([[CHAIN:%.*]]: !mef.chain{{.*}}[[GPU:%.*]]: !mgp.device_context<<"gpu", 0>>
// CHECK: [[SPEC:%.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
// CHECK: [[IDX4:%.*]] = index.constant 4
// CHECK: [[IDX16:%.*]] = index.constant 16
// CHECK: [[BUF:%.*]] = mgp.buffer.alloc [[CHAIN]], [[IDX16]], [[GPU]]
// CHECK: [[TENSOR:%.*]] = mgp.tensor.create [[CHAIN]], [[BUF]], [[SPEC]]
// CHECK: mef.output [[CHAIN]]
mo.graph @no_params() {
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>
  mo.output
}

// CHECK-LABEL: mgp.model @params
// CHECK: execute ([[CHAIN:%.*]]: !mef.chain{{.*}}[[GPU:%.*]]: !mgp.device_context<<"gpu", 0>>
// CHECK: [[D0:%.*]] = mosh.param.to_value = <d0>
// CHECK: [[D0_SI64:%.*]] = pop.cast_to_builtin [[D0]] : !kgen.scalar<si64> to si64
// CHECK: [[D0_IDX:%.*]] = index.casts [[D0_SI64]] : si64 to index
// CHECK: [[D1:%.*]] = mosh.param.to_value = <d1>
// CHECK: [[D1_SI64:%.*]] = pop.cast_to_builtin [[D1]] : !kgen.scalar<si64> to si64
// CHECK: [[D1_IDX:%.*]] = index.casts [[D1_SI64]] : si64 to index
// CHECK: [[SPEC:%.*]] = mgp.tensor_spec.create [[D0_IDX]], [[D1_IDX]]
// CHECK: [[IDX4:%.*]] = index.constant 4
// CHECK: [[MUL0:%.*]] = index.mul [[D0_IDX]], [[IDX4]]
// CHECK: [[MUL1:%.*]] = index.mul [[MUL0]], [[D1_IDX]]
// CHECK: [[BUF:%.*]] = mgp.buffer.alloc [[CHAIN]], [[MUL1]], [[GPU]]
// CHECK: [[TENSOR:%.*]] = mgp.tensor.create [[CHAIN]], [[BUF]], [[SPEC]]
// CHECK: mef.output [[CHAIN]]
mo.graph @params<d0, d1>() {
  %darg0 = mo.buffer.create : !mo.buffer<[d0, d1], f32, gpu:0>
  mo.output
}
