// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

lit.struct.decl @MyStruct {}
lit.struct.decl @MyStructParams<a, b: dtype, c: regtype> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !kgen.declref<@MyStructParams<a, :dtype b, :regtype c>>
kgen.generator @UseStruct<a, b: dtype, c: regtype>(%arg0: !kgen.declref<@MyStructParams<a, :dtype b, :regtype c>>) {
  kgen.return
}

// CHECK-LABEL: @metatype
// CHECK-SAME: !kgen.pointer<@MyStruct : metatype<@MyStruct>>
kgen.generator @metatype(%arg0: !kgen.pointer<@MyStruct : metatype<@MyStruct>>) {
  kgen.return
}

// CHECK-LABEL: @declref_metatype
// CHECK-SAME: !kgen.declref<@MyStruct, !lit.metatype<@MyStruct>>
kgen.generator @declref_metatype(%arg0: !kgen.declref<@MyStruct, !lit.metatype<@MyStruct>>) {
  kgen.return
}

// CHECK-LABEL: lit.trait.decl @TParam<MT: regtype, T: !kgen.paramref<MT>>
lit.trait.decl @TParam<MT: regtype, T: !kgen.paramref<MT>> {
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
// CHECK: !lit<type_signature<"i": variadic<index>> param_vararg>
"type.sig"() : () -> !lit<type_signature<"i": variadic<index>> param_vararg>

// CHECK-LABEL: @type_sig
// CHECK-SAME: !lit.type_signature<index, array<*(0,0), index>>
kgen.generator @type_sig(%arg0: !lit.type_signature<index, array<*(0,0), index>>) {
  kgen.return
}

kgen.generator @nested_index<a>(%arg0: !lit.type_signature<index, index = *(0,0)>) {
  kgen.return
}

kgen.generator @subst_type<T: regtype>(%arg0: !kgen.paramref<T>) {
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
  // CHECK: bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:regtype !lit.type_signature<index, index = *(0,0)>>>
  kgen.param.declare bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:regtype !lit.type_signature<index, index = *(0,0)>>>
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

lit.trait.decl @Trait {
}

kgen.generator @method() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @trait
kgen.generator @trait() {
  // CHECK-NEXT: trait<@Trait> = <@MyStructParams<1, :dtype f32, :regtype i32>>
  kgen.param.declare type: trait<@Trait> = <@MyStructParams<1, :dtype f32, :regtype i32>>
  // CHECK-NEXT: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :regtype i32>, {"method" : () -> () = @method}]>
  kgen.param.declare vtable: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :regtype i32>, {"method" : () -> () = @method}]>
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_types
// CHECK: %arg0: !lit.ref<index, #lit.lifetime, 4>
kgen.generator @ref_types(%arg0: !lit.ref<index, #lit.lifetime, 4>) {
  kgen.unreachable
}
