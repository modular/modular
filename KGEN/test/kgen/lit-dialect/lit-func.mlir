// RUN: kgen-opt %s -verify-parameters | kgen-opt -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode | kgen-opt -verify-parameters | FileCheck %s

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
  // CHECK: !lit.generator<("a": index, |, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, "b": index, "c": index) -> ()> = <@slash>
  kgen.return
}

// CHECK-LABEL: lit.func @slashOnly()
lit.func @slashOnly(|) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(|) -> ()> = <@slashOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @slashFirst(%a: index, %b: index, %c: index)
lit.func @slashFirst(|, %a: index, %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<(|, "a": index, "b": index, "c": index) -> ()> = <@slashFirst>
  kgen.return
}

// CHECK-LABEL: lit.func @slashLast(%a: index, %b: index, %c: index, |)
lit.func @slashLast(%a: index, %b: index, %c: index, |) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index, |) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, "b": index, "c": index, |) -> ()> = <@slashLast>
  kgen.return
}

// CHECK-LABEL: lit.func @star(%a: index, *, %b: index, %c: index)
lit.func @star(%a: index, *, %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, *, "b": index, "c": index) -> ()> = <@star>
  kgen.return
}

// CHECK-LABEL: lit.func @starOnly()
lit.func @starOnly(*) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(*) -> ()> = <@starOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @starFirst(*, %a: index, %b: index, %c: index)
lit.func @starFirst(*, %a: index, %b: index, %c: index) {
  // CHECK: !lit.generator<(*, "a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<(*, "a": index, "b": index, "c": index) -> ()> = <@starFirst>
  kgen.return
}

// CHECK-LABEL: lit.func @starLast(%a: index, %b: index, %c: index)
lit.func @starLast(%a: index, %b: index, %c: index, *) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, "b": index, "c": index, *) -> ()> = <@starLast>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStar(%a: index, |, %b: index, *, %c: index)
lit.func @slashAndStar(%a: index, |,  %b: index, *, %c: index) {
  // CHECK: !lit.generator<("a": index, |, "b": index, *, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, "b": index, *, "c": index) -> ()> = <@slashAndStar>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStarTogether(%a: index, |, *, %b: index, %c: index)
lit.func @slashAndStarTogether(%a: index, |, *,  %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, |, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, *, "b": index, "c": index) -> ()> = <@slashAndStarTogether>
  kgen.return
}

// CHECK-LABEL: lit.func @slashAndStarOnly()
lit.func @slashAndStarOnly(|, *) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(|, *) -> ()> = <@slashAndStarOnly>
  kgen.return
}

// CHECK-LABEL: lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index owned = 1)
lit.func @signature_type<dt: dtype, w: scalar<dt>>(%a: index owned = 1) {
  // CHECK: self: !lit.generator<<"dt": dtype, "w": scalar<*(0,0)>>("a": index owned = 1) -> ()> = <@signature_type>
  kgen.param.declare self: !lit.generator<<"dt": dtype, "w": scalar<*(0,0)>>("a": index owned = 1) -> ()> = <@signature_type>
  // CHECK: call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.generator<("a": index owned = 1) -> ()>
  kgen.call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.generator<("a": index owned = 1) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @variadic<x: variadic<index> var, y: dtype pack>[mut lt](
lit.func @variadic<x: variadic<index> var, y: dtype pack>[mut lt](
  // CHECK-SAME: %a: !kgen.variadic<index> var, %b: !kgen.variadic<index> owned|var,
  // CHECK-SAME: %c: !lit.ref<!kgen.pack<[index, index]>, mut lt> read_mem|pack)
  %a: !kgen.variadic<index> var, %b: !kgen.variadic<index> owned|var, %c: !lit.ref<!kgen.pack<[index, index]>, mut lt> read_mem|pack
) {
  kgen.return
}

// CHECK-LABEL: lit.func @default_params<
// CHECK-SAME: a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
// CHECK-SAME: d: dtype, e: dtype = si8, f: scalar<si16> = 2
// CHECK-SAME: >(%z: index owned = 42)
lit.func @default_params<
  a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
  d: dtype, e: dtype = si8, f: scalar<si16> = 2
>(%z: index owned = 42) {
  // CHECK: self: !lit.generator<
  // CHECK-SAME: <"a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  // CHECK-SAME: >("z": index owned = 42) -> ()> = <@default_params>
  kgen.param.declare self: !lit.generator<<
    "a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  >("z": index owned = 42) -> ()> = <@default_params>

  // CHECK: call @default_params<
  // CHECK-SAME: :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  // CHECK-SAME: >(%z) : !lit.generator<("z": index owned = 42) -> ()>
  kgen.call @default_params<
    :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  >(%z) : !lit.generator<("z": index owned = 42) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.func @default_args(
// CHECK-SAME: %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3)
lit.func @default_args(
  %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3
) {
  // CHECK: call @default_args(%a, %b, %c, %d, %e, %f) : !lit.generator<
  // CHECK-SAME: ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>
  kgen.call @default_args(%a, %b, %c, %d, %e, %f) : !lit.generator<
    ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.func @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index owned = 42)
lit.func @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index owned = 42) {
  // CHECK: self: !lit.generator<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index owned = 42) -> ()> = <@star_slash_params>
  kgen.param.declare self: !lit.generator<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index owned = 42) -> ()> = <@star_slash_params>
  kgen.return
}

lit.func @create_simd<x>() -> !pop.simd<x, si8> {
  kgen.unreachable
}

// CHECK-LABEL: lit.func @parametric_default_arg
// CHECK-SAME: <x>(%y: !pop.simd<x, si8> =
// CHECK-SAME: apply(:!lit.generator<() -> !pop.simd<x, si8>> @create_simd<x>))
lit.func @parametric_default_arg<x>(%y: !pop.simd<x, si8> =
    apply(:!lit.generator<() -> !pop.simd<x, si8>> @create_simd<x>)) {
  kgen.return
}

// CHECK-LABEL: lit.func @call_parametric_default_arg
lit.func @call_parametric_default_arg(%x: !pop.simd<4, si8>) {
  // CHECK: call @parametric_default_arg<4>(%x) : !lit.generator<("y": !pop.simd<4, si8> =
  // CHECK-SAME: apply(:!lit.generator<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.call @parametric_default_arg<4>(%x) : !lit.generator<("y": !pop.simd<4, si8> =
    apply(:!lit.generator<() -> !pop.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.func @parametric_default_param
// CHECK-SAME: <x, y = x>()
lit.func @parametric_default_param<x, y = x>() {
  kgen.return
}

// CHECK-LABEL: @call_default_param
lit.func @call_default_param() {
  // CHECK: ref: !lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  kgen.param.declare ref: !lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  // CHECK: bound: !lit.generator<<index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.param.declare bound: !lit.generator<<index = 1>() -> ()> = <bind_signature(
    :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  // CHECK: bound_new: !lit.generator<<"z": index = 1>() -> ()> = <bind_signature(
  // CHECK-SAME: :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.param.declare bound_new: !lit.generator<<"z": index = 1>() -> ()> = <bind_signature(
    :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.return
}

// CHECK-LABEL: @address_default
// CHECK-SAME: %p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>
lit.func @address_default[mut lt](%p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>) {
  // CHECK: ref: !lit.generator<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.param.declare ref: !lit.generator<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.return
}

// CHECK-LABEL: lit.func @inferred
// CHECK-SAME: <a: i1, b, +, c = 1, |>
lit.func @inferred<a: i1, b, +, c = 1, |>() {
  // CHECK-NEXT: !lit.generator<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()>
  kgen.param.constant: !lit.generator<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()> = <@inferred>

  // CHECK-NEXT: !lit.generator<<index, +, *, index>() -> ()> = <?>
  kgen.param.constant: !lit.generator<<index, +, *, index>() -> ()> = <?>

  // CHECK-NEXT: !lit.generator<<index, +>() -> ()> = <?>
  kgen.param.constant: !lit.generator<<index, +>() -> ()> = <?>
  kgen.return
}

// CHECK-LABEL: lit.func @different_param_name
lit.func @different_param_name() {
  // CHECK: lit.func nested_fn<["a"]param, |>()
  lit.func nested_fn<["a"]param, |>() {
    kgen.return
  }
  // CHECK: ref: !lit.generator<<"a": index, |>() -> ()> = <nested_fn>
  kgen.param.declare ref: !lit.generator<<"a": index, |>() -> ()> = <nested_fn>
  kgen.return
}

// CHECK-LABEL: lit.func @lifetime_set
lit.func @lifetime_set<set: origin.set>[mut lt]() {
  // CHECK-NEXT: f0: !lit.generator<:set:() capturing -> ()> = <?>
  kgen.param.declare f0: !lit.generator<:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f1: !lit.generator<:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f1: !lit.generator<:{mut lt}:() capturing -> ()> = <?>
  // CHECK-NEXT: f2: !lit.generator<[1]:set:() capturing -> ()> = <?>
  kgen.param.declare f2: !lit.generator<[1]:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f3: !lit.generator<[1]:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f3: !lit.generator<[1]:{mut lt}:() capturing -> ()> = <?>
  // CHECK-NEXT: f4: !lit.generator<<index>:set:() capturing -> ()> = <?>
  kgen.param.declare f4: !lit.generator<<index>:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f5: !lit.generator<<index>:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f5: !lit.generator<<index>:{mut lt}:() capturing -> ()> = <?>
  kgen.return
}

// CHECK-LABEL: lit.func @lambda_capture_lifetimes
lit.func @lambda_capture_lifetimes<set: origin.set>[mut lt]() {
  // CHECK: lit.func set_capture:set:<param>() capturing
  lit.func set_capture:set:<param>() capturing {
    kgen.return
  }
  // CHECK: lit.func lt_capture:{mut lt}:() capturing
  lit.func lt_capture:{mut lt}:() capturing {
    kgen.return
  }

  // CHECK: ref0: !lit.generator<<"param": index>:set:() capturing -> ()> = <set_capture>
  kgen.param.declare ref0: !lit.generator<<"param": index>:set:() capturing -> ()> = <set_capture>
  // CHECK: ref1: !lit.generator<:{mut lt}:() capturing -> ()> = <lt_capture>
  kgen.param.declare ref1: !lit.generator<:{mut lt}:() capturing -> ()> = <lt_capture>
  kgen.return
}
