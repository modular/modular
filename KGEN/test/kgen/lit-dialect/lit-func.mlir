// RUN: kgen-opt %s -verify-parameters | kgen-opt -verify-parameters | FileCheck %s

// CHECK-LABEL: lit.func @argNameParsing(
// CHECK-SAME: %a: index, %woof: index, %_21451[*"!451"]: index
lit.func @argNameParsing(%a: index, %b[woof]: index, %c[*"!451"]: index) {
  kgen.return
}

// CHECK-LABEL: lit.func @outer(%foo: index) {
lit.func @outer(%a[foo]: index) {
  // CHECK-NEXT: lit.func @inner(%foo_0[foo]: index, %foo_0_1[foo_0]: index) {
  lit.func @inner(%b[foo]: index, %c[foo_0]: index) {
    // CHECK-NEXT: lit.func @more_inner(%foo_2[foo]: index) {
    lit.func @more_inner(%d[foo]: index) {
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: lit.func @slash(%a: index, |, %b: index, %c: index)
lit.func @slash(%a: index, |, %b: index, %c: index) {
  // CHECK: !lit.signature<("a": index, |, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, |, "b": index, "c": index) -> ()> = <@slash>
  kgen.return
}

// CHECK-LABEL: lit.func @slashOnly()
lit.func @slashOnly(|) {
  // CHECK: !lit.signature<() -> ()>
  kgen.param.declare self: !lit.signature<(|) -> ()> = <@slashOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @slashFirst(%a: index, %b: index, %c: index)
lit.func @slashFirst(|, %a: index, %b: index, %c: index) {
  // CHECK: !lit.signature<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<(|, "a": index, "b": index, "c": index) -> ()> = <@slashFirst>
  kgen.return
}

// CHECK-LABEL: lit.func @slashLast(%a: index, %b: index, %c: index, |)
lit.func @slashLast(%a: index, %b: index, %c: index, |) {
  // CHECK: !lit.signature<("a": index, "b": index, "c": index, |) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, "b": index, "c": index, |) -> ()> = <@slashLast>
  kgen.return
}

// CHECK-LABEL: lit.func @star(%a: index, *, %b: index, %c: index)
lit.func @star(%a: index, *, %b: index, %c: index) {
  // CHECK: !lit.signature<("a": index, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, *, "b": index, "c": index) -> ()> = <@star>
  kgen.return
}

// CHECK-LABEL: lit.func @starOnly()
lit.func @starOnly(*) {
  // CHECK: !lit.signature<() -> ()>
  kgen.param.declare self: !lit.signature<(*) -> ()> = <@starOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @starFirst(*, %a: index, %b: index, %c: index)
lit.func @starFirst(*, %a: index, %b: index, %c: index) {
  // CHECK: !lit.signature<(*, "a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<(*, "a": index, "b": index, "c": index) -> ()> = <@starFirst>
  kgen.return
}

// CHECK-LABEL: lit.func @starLast(%a: index, %b: index, %c: index)
lit.func @starLast(%a: index, %b: index, %c: index, *) {
  // CHECK: !lit.signature<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, "b": index, "c": index, *) -> ()> = <@starLast>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStar(%a: index, |, %b: index, *, %c: index)
lit.func @slashAndStar(%a: index, |,  %b: index, *, %c: index) {
  // CHECK: !lit.signature<("a": index, |, "b": index, *, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, |, "b": index, *, "c": index) -> ()> = <@slashAndStar>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStarTogether(%a: index, |, *, %b: index, %c: index)
lit.func @slashAndStarTogether(%a: index, |, *,  %b: index, %c: index) {
  // CHECK: !lit.signature<("a": index, |, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.signature<("a": index, |, *, "b": index, "c": index) -> ()> = <@slashAndStarTogether>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStarOnly()
lit.func @slashAndStarOnly(|, *) {
  // CHECK: !lit.signature<() -> ()>
  kgen.param.declare self: !lit.signature<(|, *) -> ()> = <@slashAndStarOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index borrow = 1)
lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index borrow = 1) {
  // CHECK: self: !lit.signature<<"dt": dtype, "w": scalar<*(0,0)>>("a": index borrow = 1) -> ()> = <@signature_type>
  kgen.param.declare self: !lit.signature<<"dt": dtype, "w": scalar<*(0,0)>>("a": index borrow = 1) -> ()> = <@signature_type>
  // CHECK: call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.signature<("a": index borrow = 1) -> ()>
  kgen.call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.signature<("a": index borrow = 1) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @variadic<x: variadic<index> var, y: dtype pack>(
lit.func @variadic<x: variadic<index> var, y: dtype pack>(
  // CHECK-SAME: %a: !kgen.variadic<index> var, %b: !kgen.variadic<index> borrow|var, %c: !kgen.pack<[index, index]> pack
  %a: !kgen.variadic<index> var, %b: !kgen.variadic<index> borrow|var, %c: !kgen.pack<[index, index]> pack
) {
  kgen.return
}

// CHECK-LABEL: lit.func @default_params<
// CHECK-SAME: a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
// CHECK-SAME: d: dtype, e: dtype = si8, f: scalar<si16> = 2
// CHECK-SAME: >(%z: index borrow = 42)
lit.func @default_params<
  a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
  d: dtype, e: dtype = si8, f: scalar<si16> = 2
>(%z: index borrow = 42) {
  // CHECK: self: !lit.signature<
  // CHECK-SAME: <"a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  // CHECK-SAME: >("z": index borrow = 42) -> ()> = <@default_params>
  kgen.param.declare self: !lit.signature<<
    "a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  >("z": index borrow = 42) -> ()> = <@default_params>

  // CHECK: call @default_params<
  // CHECK-SAME: :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  // CHECK-SAME: >(%z) : !lit.signature<("z": index borrow = 42) -> ()>
  kgen.call @default_params<
    :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  >(%z) : !lit.signature<("z": index borrow = 42) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.func @default_args(
// CHECK-SAME: %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3)
lit.func @default_args(
  %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3
) {
  // CHECK: call @default_args(%a, %b, %c, %d, %e, %f) : !lit.signature<
  // CHECK-SAME: ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>
  kgen.call @default_args(%a, %b, %c, %d, %e, %f) : !lit.signature<
    ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.func @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index borrow = 42)
lit.func @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index borrow = 42) {
  // CHECK: self: !lit.signature<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@star_slash_params>
  kgen.param.declare self: !lit.signature<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index borrow = 42) -> ()> = <@star_slash_params>
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

// CHECK-LABEL: lit.func @call_parametric_default_arg
lit.func @call_parametric_default_arg(%x: !pop.simd<4, si8>) {
  // CHECK: call @parametric_default_arg<4>(%x) : !lit.signature<("y": !pop.simd<4, si8> =
  // CHECK-SAME: apply(:!lit.signature<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.call @parametric_default_arg<4>(%x) : !lit.signature<("y": !pop.simd<4, si8> =
    apply(:!lit.signature<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @parametric_default_param
// CHECK-SAME: <x, y = x>()
lit.func @parametric_default_param<x, y = x>() {
  kgen.return
}

// CHECK-LABEL: @call_default_param
lit.func @call_default_param() {
  // CHECK: ref: !lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  kgen.param.declare ref: !lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  // CHECK: bound: !lit.signature<<index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.param.declare bound: !lit.signature<<index = 1>() -> ()> = <bind_signature(
    :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  // CHECK: bound_new: !lit.signature<<"z": index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.param.declare bound_new: !lit.signature<<"z": index = 1>() -> ()> = <bind_signature(
    :!lit.signature<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.return
}

// CHECK-LABEL: @address_default
// CHECK-SAME: %p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>
lit.func @address_default[mut lt](%p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>) {
  // CHECK: ref: !lit.signature<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.param.declare ref: !lit.signature<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.return
}

// CHECK-LABEL: lit.func @inferred
// CHECK-SAME: <a: i1, b, +, c = 1, |>
lit.func @inferred<a: i1, b, +, c = 1, |>() {
  // CHECK-NEXT: !lit.signature<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()>
  kgen.param.constant: !lit.signature<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()> = <@inferred>

  // CHECK-NEXT: !lit.signature<<index, +, *, index>() -> ()> = <?>
  kgen.param.constant: !lit.signature<<index, +, *, index>() -> ()> = <?>

  // CHECK-NEXT: !lit.signature<<index, +>() -> ()> = <?>
  kgen.param.constant: !lit.signature<<index, +>() -> ()> = <?>
  kgen.return
}

// CHECK-LABEL: lit.func @different_param_name
lit.func @different_param_name() {
  // CHECK: lit.func nested_fn<["a"]param, |>()
  lit.func nested_fn<["a"]param, |>() {
    kgen.return
  }
  // CHECK: ref: !lit.signature<<"a": index, |>() -> ()> = <nested_fn>
  kgen.param.declare ref: !lit.signature<<"a": index, |>() -> ()> = <nested_fn>
  kgen.return
}
