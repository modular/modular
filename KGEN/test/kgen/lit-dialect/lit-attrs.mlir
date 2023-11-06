// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #lit.fn_metadata<["someRef", "v"], [pos, kw], ["someParam", "paramWithDefault"], [pos, pos_or_kw], [13 : index, 17 : i64], [3.140000e+00 : f32]>
"some.op"() {metadata = #lit.fn_metadata<["someRef", "v"], [pos, kw], ["someParam", "paramWithDefault"], [pos, pos_or_kw],  [13 : index, 17: i64], [3.14: f32]>} : () -> ()

// CHECK: #lit.fn_metadata<[], [], [], [], [], []>
"some.op"() {metadata = #lit.fn_metadata<[], [], [], [], [], []>} : () -> ()

// CHECK: #kgen.none : !kgen.none
"a"() {a = #kgen.none : !kgen.none} : () -> ()

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
