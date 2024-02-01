// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: "arg_param_list.with_defaults"
// CHECK-SAME: {arg_param_list = #lit.arg_param_list<
// CHECK-SAME: ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [4.200000e+00 : f32], [1 : i64], [], []>}
"arg_param_list.with_defaults"() {arg_param_list = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [4.2 : f32], [1: i64], [], []
>} : () -> ()

// CHECK: "arg_param_list.with_variadics"
// CHECK-SAME: {arg_param_list = #lit.arg_param_list<
// CHECK-SAME: ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [], [], [3], [1]>}
"arg_param_list.with_variadics"() {arg_param_list = #lit.arg_param_list<
  ["a", "b", "c", "d"], [pos, pos_or_kw, kw, kw], [], [], [3], [1]
>} : () -> ()

// CHECK: "empty.arg_param_list"
// CHECK-SAME: {arg_param_list = #lit.arg_param_list<[], [], [], [], [], []>}
"empty.arg_param_list"() {arg_param_list = #lit.arg_param_list<[], [], [], [], [], []>} : () -> ()

// CHECK: #lit.fn_metadata
// CHECK-SAME: <["someRef", "v"], [pos, kw], [13 : index], [17 : i64], [], []>,
// CHECK-SAME: <["someParam", "paramWithDefault"], [pos, pos_or_kw], [], [], [1], []>,
// CHECK-SAME: 2>
"some.op"() {metadata = #lit.fn_metadata<
  <["someRef", "v"], [pos, kw], [13 : index], [17 : i64], [], []>,
  <["someParam", "paramWithDefault"], [pos, pos_or_kw], [], [], [1], []>,
  2
>} : () -> ()

// CHECK: #lit.fn_metadata<<[], [], [], [], [], []>, <[], [], [], [], [], []>, 0>
"some.op"() {metadata = #lit.fn_metadata<<[], [], [], [], [], []>, <[], [], [], [], [], []>, 0>} : () -> ()

// CHECK: #kgen.none : !kgen.none
"a"() {a = #kgen.none : !kgen.none} : () -> ()

// CHECK: #lit.type_lineage<index, [index]>
"a"() {a = #lit.type_lineage<index, [index]>} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK: #lit.struct<{foo = 5, bar: dtype = f32}>
"a"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !kgen.declref<@Foo>} : () -> ()

// CHECK: #lit.lifetime : !lit.lifetime<1>
"a"() {a = #lit.lifetime : !lit.lifetime<1>} : () -> ()


kgen.generator @lifetime_lower<p: !lit.lifetime<1>>(%a: !lit.lifetime<0>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK: %lifetime = kgen.param.constant: lifetime<0> = <#lit.lifetime>
  %cst = kgen.param.constant: lifetime<0> = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower<:lifetime<1> #lit.lifetime>(%lifetime) : (!lit.lifetime<0>) -> ()
  kgen.call @lifetime_lower<:lifetime<1> #lit.lifetime>(%cst) : (!lit.lifetime<0>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: lifetime<0>, q: lifetime<1>>(
// CHECK-SAME: %arg0: !lit.ref<@Foo, imm p>
// CHECK-SAME: %arg1: !lit.ref<@Foo, mut q>)
kgen.generator @ref_type<p: !lit.lifetime<0>, q: !lit.lifetime<1>>
(%a: !lit.ref<@Foo, imm p>, %b: !lit.ref<@Foo, mut q>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @metatype
kgen.generator @metatype() {
  // CHECK: T: metatype<@Foo> = <@Foo>
  kgen.param.declare T: metatype<@Foo> = <@Foo>
  kgen.return
}

lit.struct.decl @Bar<a, b: dtype> {}
lit.struct.decl @BarDefaults<a, b: dtype = f32> {}

// CHECK-LABEL: kgen.generator @bind_type
kgen.generator @bind_type<T: metatype<@Bar<?, :dtype ?>, <index, dtype>>>() {
  // CHECK: FullyBound: metatype<@Bar<16, :dtype f32>> =
  // CHECK-SAME: #lit.bind_type<:metatype<@Bar<?, :dtype ?>, <index, dtype>> T, [16, f32]>
  kgen.param.declare FullyBound: metatype<@Bar<16, :dtype f32>> = <
    #lit.bind_type<
      :metatype<@Bar<?, :dtype ?>, <index, dtype>> T,
      [16, f32]
    >
  >

  // CHECK: PartiallyBound: metatype<@Bar<?, :dtype f32>, <index>> =
  // CHECK-SAME: #lit.bind_type<:metatype<@Bar<?, :dtype ?>, <index, dtype>> T, [?, f32]>
  kgen.param.declare PartiallyBound: metatype<@Bar<?, :dtype f32>, <index>> = <
    #lit.bind_type<
      :metatype<@Bar<?, :dtype ?>, <index, dtype>> T,
      [?, f32]
    >
  >

  // CHECK: PartiallyBoundDefaults: metatype<@BarDefaults<16, :dtype ?>, <dtype = f32>> =
  // CHECK-SAME: #lit.bind_type<:metatype<@BarDefaults<?, :dtype ?>, <index, dtype = f32>> ?, [16, ?]>
  kgen.param.declare PartiallyBoundDefaults: metatype<@BarDefaults<16, :dtype ?>, <dtype = f32>> = <
    #lit.bind_type<
      :metatype<@BarDefaults<?, :dtype ?>, <index, dtype = f32>> ?,
      [16, ?]
    >
  >

  // CHECK: BoundDeclRef: metatype<@Bar<?, :dtype f32>, <index>> =
  // CHECK-SAME: <@Bar<?, :dtype f32>>
  kgen.param.declare BoundDeclRef: metatype<@Bar<?, :dtype f32>, <index>> = <
    #lit.bind_type<
      :metatype<@Bar<?, :dtype ?>, <index, dtype>> @Bar<?, :dtype ?>,
      [?, f32]
    >
  >

  // CHECK: BoundFromPartial: metatype<@Bar<16, :dtype f32>> =
  // CHECK-SAME: #lit.bind_type<:metatype<@Bar<?, :dtype f32>, <index>> ?, [16]>
  kgen.param.declare BoundFromPartial: metatype<@Bar<16, :dtype f32>> = <
    #lit.bind_type<
      :metatype<@Bar<?, :dtype f32>, <index>> ?,
      [16]
    >
  >

  kgen.return
}

// CHECK-LABEL: kgen.generator @unpacked
kgen.generator @unpacked<T: type>() {
  // CHECK: kgen.param.constant: !lit.unpacked<T> = <#lit.unpacked<?>>
  %c = kgen.param.constant: !lit.unpacked<T> = <#lit.unpacked<?>>
  kgen.return
}

// CHECK-LABEL: @lifetime_union
kgen.generator @lifetime_union<x: !lit.lifetime<0>, y: !lit.lifetime<0>>() {
  // CHECK-NEXT: %a = lit.varlet.decl
  %a = lit.varlet.decl "a" imp : !lit.ref<index, mut z>

  // CHECK-NEXT: "a"() {a = #lit.lifetime : !lit.lifetime<0>} : () -> ()
  "a"() {a = #lit.lifetime.union<#lit.lifetime : !lit.lifetime<0>> : !lit.lifetime<0>} : () -> ()
  // CHECK-NEXT: "b"() {a = #kgen.param.decl.ref<"x"> : !lit.lifetime<0>}
  "b"() {a = #lit.lifetime.union<#lit.lifetime : !lit.lifetime<0>,
                                 #kgen.param.decl.ref<"x"> :!lit.lifetime<0>>
        : !lit.lifetime<0>} : () -> ()
  // CHECK-NEXT: "c"() {a = #lit.lifetime.union<#kgen.param.decl.ref<"x"> : !lit.lifetime<0>, #kgen.param.decl.ref<"y"> : !lit.lifetime<0>> : !lit.lifetime<0>}
  "c"() {a = #lit.lifetime.union<#lit.lifetime : !lit.lifetime<0>,
                                 #kgen.param.decl.ref<"x"> :!lit.lifetime<0>,
                                 #kgen.param.decl.ref<"y"> :!lit.lifetime<0>>
        : !lit.lifetime<0>} : () -> ()

  // CHECK-NEXT: kgen.param.declare nothing: lifetime<0> = <#lit.lifetime>
  kgen.param.declare nothing: !lit.lifetime<0> = <#lit.lifetime>
  // CHECK-NEXT:  kgen.param.declare nothing_2: lifetime<0> = <#lit.lifetime>
  kgen.param.declare nothing_2: !lit.lifetime<0> = <{#lit.lifetime, #lit.lifetime}>
  // CHECK-NEXT: kgen.param.declare x_ref: lifetime<0> = <x>
  kgen.param.declare x_ref: !lit.lifetime<0> = <x>
  // CHECK-NEXT: kgen.param.declare x_ref2: lifetime<0> = <x>
  kgen.param.declare x_ref2: !lit.lifetime<0> = <*"x">
  // CHECK-NEXT: kgen.param.declare x_or_y_ref: lifetime<0> = <{x, y}>
  kgen.param.declare x_or_y_ref: !lit.lifetime<0> = <{x, y, x}>
  // CHECK-NEXT: kgen.param.declare y_ref: lifetime<0> = <y>
  kgen.param.declare y_ref: !lit.lifetime<0> = <{y, #lit.lifetime}>
  // CHECK-NEXT: kgen.param.declare xyz_ref: lifetime<0> = <{x, y, (mutcast mut z)}>
  kgen.param.declare xyz_ref: !lit.lifetime<0> = <{{x, y}, {(mutcast mut z), y}}>

  kgen.return
}
