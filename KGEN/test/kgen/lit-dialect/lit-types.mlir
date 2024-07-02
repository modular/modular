// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

lit.struct.decl @MyStruct {}
lit.struct.decl @MyStructParams<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !lit.declref<@MyStructParams<a, :dtype b, :type c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !lit.declref<@MyStructParams<a, :dtype b, :type c>>) {
  kgen.return
}

// CHECK-LABEL: @metatype
// CHECK-SAME: !kgen.pointer<@MyStruct : type>
kgen.generator @metatype(%arg0: !kgen.pointer<@MyStruct : type>) {
  kgen.return
}

// CHECK-LABEL: @declref_metatype
// CHECK-SAME: !lit.declref<@MyStruct, type>
kgen.generator @declref_metatype(%arg0: !lit.declref<@MyStruct, type>) {
  kgen.return
}

// CHECK-LABEL: lit.trait.decl @TParam<MT: type, T: !kgen.paramref<MT>>
lit.trait.decl @TParam<MT: type, T: !kgen.paramref<MT>> {
  // CHECK-NEXT: lit.func @f(%self: !kgen.paramref<:!kgen.paramref<MT> T>) -> !kgen.none
  lit.func @f(%self: !kgen.paramref<:!kgen.paramref<MT> T>) -> !kgen.none {
    lit.trait_func
  }
}

lit.trait.decl @MyTrait {}

// CHECK-LABEL: @trait_metatype
// CHECK-SAME: !kgen.paramref<:trait<@MyTrait> T>
lit.func @trait_metatype<T: trait<@MyTrait>>(%arg0: !kgen.paramref<:trait<@MyTrait> T>) {
  kgen.return
}

// CHECK: !lit.type_signature
"type.sig"() : () -> !lit.type_signature
// CHECK: !lit.type_signature<index, |>
"type.sig"() : () -> !lit.type_signature<index, |>
// CHECK: !lit.type_signature<"dt": dtype = f32>
"type.sig"() : () -> !lit.type_signature<"dt": dtype = f32>
// CHECK: !lit.type_signature<"i": variadic<index> var>
"type.sig"() : () -> !lit<type_signature<"i": variadic<index> var>>

// CHECK-LABEL: @type_sig
// CHECK-SAME: !lit.type_signature<index, array<*(0,0), index>>
kgen.generator @type_sig(%arg0: !lit.type_signature<index, array<*(0,0), index>>) {
  kgen.return
}

kgen.generator @nested_index<a>(%arg0: !lit.type_signature<index, index = *(0,0)>) {
  kgen.return
}

kgen.generator @subst_type<T: type>(%arg0: !kgen.paramref<T>) {
  kgen.return
}

kgen.generator @return_type() -> !lit.type_signature<index, index = *(0,0)> {
  kgen.unreachable
}

kgen.generator @return_sig() -> !lit.signature<<index, index = *(0,0)>() -> ()> {
  kgen.unreachable
}

// CHECK-LABEL: @bind_nested
kgen.generator @bind_nested() {
  // CHECK: bound0: (!lit.type_signature<index, index = *(0,0)>) -> () = <@nested_index<1>>
  kgen.param.declare bound0: (!lit.type_signature<index, index = *(0,0)>) -> () = <@nested_index<1>>
  // CHECK: bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:type !lit.type_signature<index, index = *(0,0)>>>
  kgen.param.declare bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:type !lit.type_signature<index, index = *(0,0)>>>
  // CHECK: result0: !lit.type_signature<index, index = *(0,0)> = <apply(:() -> !lit.type_signature<index, index = *(0,0)> @return_type)>
  kgen.param.declare result0: !lit.type_signature<index, index = *(0,0)> = <apply(:() -> !lit.type_signature<index, index = *(0,0)> @return_type)>
  // CHECK: result1: !lit.signature<<index, index = *(0,0)>() -> ()> = <apply(:() -> !kgen.signature<!lit.signature<<index, index = *(0,0)>() -> ()>> @return_sig)>
  kgen.param.declare result1: !lit.signature<<index, index = *(0,0)>() -> ()> = <apply(:() -> !lit.signature<<index, index = *(0,0)>() -> ()> @return_sig)>
  kgen.return
}

// CHECK-LABEL: @passing_kinds
kgen.generator @passing_kinds(
    // CHECK-SAME: !lit.signature<<i8, |, i8, *, i8, ?, i8>() -> ()>
    %arg0: !lit.signature<<i8, |, i8, *, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.signature<<i8, i8, *, i8, ?, i8>() -> ()>
    %arg1: !lit.signature<<i8, i8, *, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.signature<<i8, |, i8, i8, ?, i8>() -> ()>
    %arg2: !lit.signature<<i8, |, i8, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.signature<<i8, i8, i8, ?, i8>() -> ()>
    %arg3: !lit.signature<<i8, i8, i8, ?, i8>() -> ()>
) {
  kgen.return
}

// Test that passing kind printing/parsing works correctly for long signatures.
!t = !kgen.signature<!lit.signature<(
    "a": index owned, |, *,
    "b": index owned, "c": index owned, "d": index owned, "e": index owned,
    "f": index owned, "g": index owned, "h": index owned, "i": index owned,
    "j": index owned, "k": index owned, "l": index owned, "m": index owned
) -> !kgen.none>>

// CHECK-LABEL: lit.func @long_sig(%t: !kgen.signature<!lit.signature<
// CHECK-SAME: "a": index owned, |, *,
// CHECK-SAME: "b": index owned, "c": index owned, "d": index owned, "e": index owned,
// CHECK-SAME: "f": index owned, "g": index owned, "h": index owned, "i": index owned,
// CHECK-SAME: "j": index owned, "k": index owned, "l": index owned, "m": index owned
lit.func @long_sig(%t: !t) {
    kgen.return
}

lit.trait.decl @Trait {
}

kgen.generator @method() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @trait
kgen.generator @trait() {
  // CHECK-NEXT: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  kgen.param.declare type: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  // CHECK-NEXT: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, {"method" : () -> () = @method}]>
  kgen.param.declare vtable: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, {"method" : () -> () = @method}]>
  // CHECK-NEXT: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  kgen.param.declare type_same: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<1, :dtype f32, :type i32>]>
  // CHECK-NEXT: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>]>
  kgen.param.declare type_diff: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>]>
  // CHECK-NEXT: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>, {"method" : () -> () = @method}]>
  kgen.param.declare vtable_diff: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>, {"method" : () -> () = @method}]>
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_types
// CHECK: %arg0: !lit.ref<index, imm #lit.lifetime, 4>
kgen.generator @ref_types(%arg0: !lit.ref<index, imm #lit.lifetime, 4>) {
  kgen.unreachable
}

// FIXME: This isn't valid IR!
// CHECK-LABEL: kgen.generator @escape_meta_type_param_names<type: anystruct<@Trait>>()
kgen.generator @escape_meta_type_param_names<type: anystruct<@Trait>>() {
  // CHECK: kgen.param.declare T: anystruct<@Trait> = <*"type">
  kgen.param.declare T: anystruct<@Trait> = <*"type">
  kgen.return
}

// CHECK-LABEL: kgen.generator @escape_trait_param_names<type: trait<@Trait>>()
kgen.generator @escape_trait_param_names<type: trait<@Trait>>() {
  // CHECK: kgen.param.declare T: trait<@Trait> = <*"type">
  kgen.param.declare T: trait<@Trait> = <*"type">
  kgen.return
}
