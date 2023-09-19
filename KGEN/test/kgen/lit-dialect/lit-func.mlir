// RUN: kgen-opt %s -verify-parameters | FileCheck %s

// CHECK: lit.func @positional_args(%a: index, %b: index) numPosArgs(1)
lit.func @positional_args(%a: index, %b: index) numPosArgs(1) {
  // CHECK: self: !lit.signature<(index, "b": index) -> ()> = <@positional_args>
  kgen.param.declare self: !lit.signature<(index, "b": index) -> ()> = <@positional_args>
  kgen.return
}

// CHECK-LABEL: lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index borrow = 1)
lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index borrow = 1) {
  // CHECK: self: !lit.signature<<dtype, scalar<*(0,0)>>("a": index borrow = 1) -> ()> = <@signature_type>
  kgen.param.declare self: !lit.signature<<dtype, scalar<*(0,0)>>("a": index borrow = 1) -> ()> = <@signature_type>
  // CHECK: call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.signature<("a": index borrow = 1) -> ()>
  kgen.call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.signature<("a": index borrow = 1) -> ()>
  kgen.return
}
