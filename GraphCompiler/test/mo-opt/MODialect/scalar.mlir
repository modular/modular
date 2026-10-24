// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK: mo.graph @scalar_int_type() {
mo.graph @scalar_int_type() {
  %0 = mo.constant.scalar <1 : si64> : !mo.scalar<si64>
  mo.output
}

// CHECK: mo.graph @scalar_bool_type() {
mo.graph @scalar_bool_type() {
  %0 = mo.constant.scalar <1 : i1> : !mo.scalar<bool>
  mo.output
}

// CHECK: mo.graph @scalar_float_type() {
mo.graph @scalar_float_type() {
  %0 = mo.constant.scalar <2.5 : f32> : !mo.scalar<f32>
  mo.output
}
