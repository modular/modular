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

// CHECK-LABEL: lit.func @default_params<_1x3_a[a]: dtype, _1x6_b[b]: dtype = f32, _1x9_w[w]: scalar<si32> = 1>(%z: index borrow = 42)
lit.func @default_params<_1x3_a[a]: dtype, _1x6_b[b]: dtype = f32, _1x9_w[w]: scalar<si32> = 1>(%z: index borrow = 42) {
  // CHECK: self: !lit.signature<<"a": dtype, "b": dtype = f32, "w": scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@default_params>
  kgen.param.declare self: !lit.signature<<"a": dtype, "b": dtype = f32, "w": scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@default_params>
  // CHECK: call @default_params<:dtype si16, :dtype f16, :scalar<si32> 5>(%z) : !lit.signature<("z": index borrow = 42) -> ()>
  kgen.call @default_params<:dtype si16, :dtype f16, :scalar<si32> 5>(%z) : !lit.signature<("z": index borrow = 42) -> ()>
  kgen.return
}

lit.func @create_simd<x>() -> !pop.simd<x, si8> {
  kgen.unreachable
}

// CHECK-LABEL: lit.func @parametric_default_arg
// CHECK-SAME: <x>(%y: !pop.simd<x, si8> =
// CHECK-SAME: apply(:!lit.signature<() -> !pop.simd<x, si8>> @create_simd<x>))
lit.func @parametric_default_arg<x>(%y: !pop.simd<x, si8> =
    apply(:!lit.signature<() -> !pop.simd<x, si8>> @create_simd<x>)) {
  kgen.return
}

// CHECK-LABEL: lit.func @call_default_arg
lit.func @call_default_arg(%x: !pop.simd<4, si8>) {
  // CHECK: call @parametric_default_arg<4>(%x) : !lit.signature<("y": !pop.simd<4, si8> =
  // CHECK-SAME: apply(:!lit.signature<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.call @parametric_default_arg<4>(%x) : !lit.signature<("y": !pop.simd<4, si8> =
    apply(:!lit.signature<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @parametric_default_param
// CHECK-SAME: <_1x3_x[x], _1x6_y[y] = _1x3_x>()
lit.func @parametric_default_param<_1x3_x[x], _1x6_y[y] = _1x3_x>() {
  kgen.return
}

// CHECK-LABEL: @call_default_param
lit.func @call_default_param() {
  // CHECK: ref: !lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  kgen.param.declare ref: !lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  // CHECK: bound: !lit.signature<<index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, #kgen.unbound)>
  kgen.param.declare bound: !lit.signature<<index = 1>() -> ()> = <bind_signature(
    :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, #kgen.unbound)>
  // CHECK: bound_new: !lit.signature<<"z": index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, #kgen.unbound)>
  kgen.param.declare bound_new: !lit.signature<<"z": index = 1>() -> ()> = <bind_signature(
    :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, #kgen.unbound)>
  kgen.return
}
