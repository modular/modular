// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #lit.fn_metadata<["someRef", "v"], [pos, kw], ["someParam", "paramWithDefault"], [pos, pos_or_kw], [13 : index, 17 : i64], [3.140000e+00 : f32], 2>
"some.op"() {metadata = #lit.fn_metadata<["someRef", "v"], [pos, kw], ["someParam", "paramWithDefault"], [pos, pos_or_kw],  [13 : index, 17: i64], [3.14: f32], 2>} : () -> ()

// CHECK: #lit.fn_metadata<[], [], [], [], [], [], 0>
"some.op"() {metadata = #lit.fn_metadata<[], [], [], [], [], [], 0>} : () -> ()

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

// CHECK: #lit.lifetime : !lit.lifetime
"a"() {a = #lit.lifetime : !lit.lifetime} : () -> ()


kgen.generator @lifetime_lower<p: !lit.lifetime>(%a: !lit.lifetime) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK: %lifetime = kgen.param.constant: lifetime = <#lit.lifetime>
  %cst = kgen.param.constant: lifetime = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%lifetime) : (!lit.lifetime) -> ()
  kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%cst) : (!lit.lifetime) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: lifetime>(
// CHECK-SAME: %arg0: !lit.ref<@Foo, p>
// CHECK-SAME: %arg1: !lit.ref<mut @Foo, p>)
kgen.generator @ref_type<p: !lit.lifetime>(%a: !lit.ref<@Foo, p>,
                                           %b: !lit.ref<mut @Foo, p>) {
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
kgen.generator @unpacked<T: regtype>() {
  // CHECK: kgen.param.constant: !lit.unpacked<T> = <#lit.unpacked<?>>
  %c = kgen.param.constant: !lit.unpacked<T> = <#lit.unpacked<?>>
  kgen.return
}

// CHECK-LABEL: @lifetime_union
kgen.generator @lifetime_union<x: !lit.lifetime, y: !lit.lifetime>() {
  // CHECK-NEXT: %a = lit.varlet.decl
  %a = lit.varlet.decl "a" imp : !lit.ref<mut index, z>

  // CHECK-NEXT: "a"() {a = #lit.lifetime : !lit.lifetime} : () -> ()
  "a"() {a = #lit.lifetime.union<#lit.lifetime> : !lit.lifetime} : () -> ()
  // CHECK-NEXT: "b"() {a = #kgen.param.decl.ref<"x"> : !lit.lifetime}
  "b"() {a = #lit.lifetime.union<#lit.lifetime,
                                 #kgen.param.decl.ref<"x"> :!lit.lifetime>
        : !lit.lifetime} : () -> ()
  // CHECK-NEXT: "c"() {a = #lit.lifetime.union<#kgen.param.decl.ref<"x"> : !lit.lifetime, #kgen.param.decl.ref<"y"> : !lit.lifetime> : !lit.lifetime} : () -> ()
  "c"() {a = #lit.lifetime.union<#lit.lifetime,
                                 #kgen.param.decl.ref<"x"> :!lit.lifetime,
                                 #kgen.param.decl.ref<"y"> :!lit.lifetime>
        : !lit.lifetime} : () -> ()

  // CHECK-NEXT: kgen.param.declare nothing: lifetime = <#lit.lifetime>
  kgen.param.declare nothing: !lit.lifetime = <#lit.lifetime>
  // CHECK-NEXT:  kgen.param.declare nothing_2: lifetime = <#lit.lifetime>
  kgen.param.declare nothing_2: !lit.lifetime = <{#lit.lifetime, #lit.lifetime}>
  // CHECK-NEXT: kgen.param.declare x_ref: lifetime = <x>
  kgen.param.declare x_ref: !lit.lifetime = <x>
  // CHECK-NEXT: kgen.param.declare x_ref2: lifetime = <x>
  kgen.param.declare x_ref2: !lit.lifetime = <*"x">
  // CHECK-NEXT: kgen.param.declare x_or_y_ref: lifetime = <{x, y}>
  kgen.param.declare x_or_y_ref: !lit.lifetime = <{x, y, x}>
  // CHECK-NEXT: kgen.param.declare y_ref: lifetime = <y>
  kgen.param.declare y_ref: !lit.lifetime = <{y, #lit.lifetime}>
  // CHECK-NEXT: kgen.param.declare xyz_ref: lifetime = <{x, y, z}>
  kgen.param.declare xyz_ref: !lit.lifetime = <{{x, y}, {z, y}}>

  kgen.return
}
