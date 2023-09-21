// RUN: kgen-opt %s -verify-parameters | kgen-opt -verify-parameters | FileCheck %s

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

// CHECK-LABEL: lit.func @default_params<a: dtype, b: dtype = f32, w: scalar<si32> = 1>(%z: index borrow = 42)
lit.func @default_params<a: dtype, b: dtype = f32, w: scalar<si32> = 1>(%z: index borrow = 42) {
  // CHECK: self: !lit.signature<<dtype, dtype = f32, scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@default_params>
  kgen.param.declare self: !lit.signature<<dtype, dtype = f32, scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@default_params>
  // CHECK: call @default_params<:dtype si16, :dtype f16, :scalar<si32> 5>(%z) : !lit.signature<("z": index borrow = 42) -> ()>
  kgen.call @default_params<:dtype si16, :dtype f16, :scalar<si32> 5>(%z) : !lit.signature<("z": index borrow = 42) -> ()>
  kgen.return
}
