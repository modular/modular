// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: "pog.metadata"
// CHECK-SAME: pog1 = #lit.pog_metadata<"some_keyword_param", pos_or_kw, false>,
// CHECK-SAME: pog2 = #lit.pog_metadata<"some_variadic_param", pos_or_kw, true>
"pog.metadata"() {
  pog1 = #lit.pog_metadata<"some_keyword_param", pos_or_kw, false>,
  pog2 = #lit.pog_metadata<"some_variadic_param", pos_or_kw, true>
} : () -> ()

// CHECK-LABEL: "pogs.with_defaults"
// CHECK-SAME: {pogs = #lit.pog_list<
// CHECK-SAME: [<"a", pos, false>, <"b", pos_or_kw, false>, <"c", kw, false>, <"d", kw, false>],
// CHECK-SAME: [4.200000e+00 : f32], [1 : i64]>}
"pogs.with_defaults"() {pogs = #lit.pog_list<
  [<"a", pos, false>, <"b", pos_or_kw, false>, <"c", kw, false>, <"d", kw, false>],
  [4.2 : f32], [1: i64]
>} : () -> ()

// CHECK-LABEL: "pogs.with_variadics"
// CHECK-SAME: {pogs = #lit.pog_list<
// CHECK-SAME: [<"a", pos, false>, <"b", pos_or_kw, false>, <"c", kw, false>, <"d", kw, true>],
// CHECK-SAME: [], [], 1, owned_in_mem>}
"pogs.with_variadics"() {pogs = #lit.pog_list<
  [<"a", pos, false>, <"b", pos_or_kw, false>, <"c", kw, false>, <"d", kw, true>],
  [], [], 1, owned_in_mem
>} : () -> ()

// CHECK-LABEL: "empty.pogs"
// CHECK-SAME: {pogs = #lit.pog_list<[], [], []>}
"empty.pogs"() {pogs = #lit.pog_list<[], [], []>} : () -> ()

// CHECK-LABEL: "some.metadata"
// CHECK-SAME: #lit.fn_metadata
// CHECK-SAME: <[<"someRef", pos, false>, <"v", kw, false>], [13 : index], [17 : i64]>,
// CHECK-SAME: <[<"someParam", pos, false>, <"paramWithDefault", pos_or_kw, true>], [], []>,
// CHECK-SAME: 2>
"some.metadata"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, false>, <"v", kw, false>], [13 : index], [17 : i64]>,
  <[<"someParam", pos, false>, <"paramWithDefault", pos_or_kw, true>], [], []>,
  2
>} : () -> ()

// CHECK-LABEL: "empty.metadata"
// CHECK-SAME: #lit.fn_metadata<<[], [], []>, <[], [], []>, 0>
"empty.metadata"() {metadata = #lit.fn_metadata<<[], [], []>, <[], [], []>, 0>} : () -> ()

// CHECK-LABEL: "none.type"
// CHECK-SAME: #kgen.none : !kgen.none
"none.type"() {a = #kgen.none : !kgen.none} : () -> ()

// CHECK-LABEL: "type_lineage.attr"
// CHECK-SAME: #lit.type_lineage<index, [index]>
"type_lineage.attr"() {a = #lit.type_lineage<index, [index]>} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK-LABEL: "struct.attr"
// CHECK-SAME: #lit.struct<{foo = 5, bar: dtype = f32}>
"struct.attr"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !lit.struct<@Foo>} : () -> ()

// CHECK-LABEL: "lifetime.attr"
// CHECK: #lit.lifetime : !lit.lifetime<1>
"lifetime.attr"() {a = #lit.lifetime : !lit.lifetime<1>} : () -> ()


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

// CHECK-LABEL: kgen.generator @anystruct
kgen.generator @anystruct() {
  // CHECK: T: anystruct<@Foo> = <@Foo>
  kgen.param.declare T: anystruct<@Foo> = <@Foo>
  kgen.return
}

lit.struct.decl @Bar<a, b: dtype> {}
lit.struct.decl @BarDefaults<a, b: dtype = f32> {}

// CHECK-LABEL: kgen.generator @bind_type
kgen.generator @bind_type<T: anystruct<@Bar<?, :dtype ?>, <index, dtype>>>() {
  // CHECK: FullyBound: anystruct<@Bar<16, :dtype f32>> =
  // CHECK-SAME: #lit.bind_type<:anystruct<@Bar<?, :dtype ?>, <index, dtype>> T, [16, f32]>
  kgen.param.declare FullyBound: anystruct<@Bar<16, :dtype f32>> = <
    #lit.bind_type<
      :anystruct<@Bar<?, :dtype ?>, <index, dtype>> T,
      [16, f32]
    >
  >

  // CHECK: PartiallyBound: anystruct<@Bar<?, :dtype f32>, <index>> =
  // CHECK-SAME: #lit.bind_type<:anystruct<@Bar<?, :dtype ?>, <index, dtype>> T, [?, f32]>
  kgen.param.declare PartiallyBound: anystruct<@Bar<?, :dtype f32>, <index>> = <
    #lit.bind_type<
      :anystruct<@Bar<?, :dtype ?>, <index, dtype>> T,
      [?, f32]
    >
  >

  // CHECK: PartiallyBoundDefaults: anystruct<@BarDefaults<16, :dtype ?>, <dtype = f32>> =
  // CHECK-SAME: #lit.bind_type<:anystruct<@BarDefaults<?, :dtype ?>, <index, dtype = f32>> ?, [16, ?]>
  kgen.param.declare PartiallyBoundDefaults: anystruct<@BarDefaults<16, :dtype ?>, <dtype = f32>> = <
    #lit.bind_type<
      :anystruct<@BarDefaults<?, :dtype ?>, <index, dtype = f32>> ?,
      [16, ?]
    >
  >

  // CHECK: BoundDeclRef: anystruct<@Bar<?, :dtype f32>, <index>> =
  // CHECK-SAME: <@Bar<?, :dtype f32>>
  kgen.param.declare BoundDeclRef: anystruct<@Bar<?, :dtype f32>, <index>> = <
    #lit.bind_type<
      :anystruct<@Bar<?, :dtype ?>, <index, dtype>> @Bar<?, :dtype ?>,
      [?, f32]
    >
  >

  // CHECK: BoundFromPartial: anystruct<@Bar<16, :dtype f32>> =
  // CHECK-SAME: #lit.bind_type<:anystruct<@Bar<?, :dtype f32>, <index>> ?, [16]>
  kgen.param.declare BoundFromPartial: anystruct<@Bar<16, :dtype f32>> = <
    #lit.bind_type<
      :anystruct<@Bar<?, :dtype f32>, <index>> ?,
      [16]
    >
  >

  kgen.return
}

// CHECK-LABEL: kgen.generator @unpacked
kgen.generator @unpacked<T: type>() {
  // CHECK: kgen.param.constant: !lit.unpacked = <#lit.unpacked>
  %c = kgen.param.constant: !lit.unpacked = <#lit.unpacked>
  kgen.return
}

// CHECK-LABEL: @lifetime_union
kgen.generator @lifetime_union<x: !lit.lifetime<0>, y: !lit.lifetime<0>>() {
  // CHECK-NEXT: %a = lit.var.decl
  %a = lit.var.decl "a" imp : !lit.ref<index, mut z>

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

  // CHECK-NEXT: "d"() {a = #lit.lifetime.union<#lit.lifetime_ref<0, 1> : !lit.lifetime<0>, #lit.lifetime_ref<1, 0> : !lit.lifetime<0>> : !lit.lifetime<0>}
  "d"() {a = #lit.lifetime.union<#lit.lifetime_ref<1, 0> : !lit.lifetime<0>, #lit.lifetime_ref<0, 1> : !lit.lifetime<0>> : !lit.lifetime<0>} : () -> ()

  kgen.param.declare is_mut: i1 = <0>
  kgen.param.declare a: lifetime<1> = <?>
  kgen.param.declare b: lifetime<0> = <?>
  kgen.param.declare c: lifetime<is_mut> = <?>

  // CHECK: <{(is_mut) c, mut a, imm b}>
  kgen.param.constant: lifetime.set = <{imm b, mut a, (is_mut) c, mut a}>
  // CHECK-NEXT: <{mut a}>
  kgen.param.constant: lifetime.set = <{imm (mutcast mut a)}>
  // CHECK-NEXT: <{}>
  kgen.param.constant: lifetime.set = <{mut #lit.lifetime, imm #lit.invalid.ref.lifetime}>
  // CHECK-NEXT: <{mut a, imm b}>
  kgen.param.constant: lifetime.set = <{mut {(mutcast imm b), a}}>

  // CHECK-NEXT: <{(mutcast mut a), b}>
  kgen.param.constant: lifetime<0> = <#lit.lifetime.set.union<#lit.lifetime.set<{mut a, imm b}> : !lit.lifetime.set>>

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
