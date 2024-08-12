// RUN: kgen-opt -lower-calling-conventions %s | FileCheck %s

// CHECK-LABEL: kgen.func @op
kgen.func @op() {
  // CHECK: "custom.my_op"() {_op_impl_params = #kgen.preserved<si32>} : () -> ()
  "custom.my_op"() {_op_impl_params = si32} : () -> ()
  // CHECK: index.constant {_op_impl_params = si32} 2
  index.constant {_op_impl_params = si32} 2
  kgen.return
}
