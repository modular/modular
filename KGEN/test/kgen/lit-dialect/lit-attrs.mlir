// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-DAG: #[[LOC1:.+]] = loc("test.mojo":5:10)
#loc1 = loc("test.mojo":5:10)
// CHECK-DAG: #[[LOC2:.+]] = loc("test.mojo":6:15)
#loc2 = loc("test.mojo":6:15)

// CHECK-LABEL: "pog.metadata"
// CHECK-SAME: pog1 = #lit.pog_metadata<"some_keyword_param", pos_or_kw, not_vararg>,
// CHECK-SAME: pog2 = #lit.pog_metadata<"some_variadic_param", pos_or_kw, pos_vararg>
"pog.metadata"() {
  pog1 = #lit.pog_metadata<"some_keyword_param", pos_or_kw, not_vararg>,
  pog2 = #lit.pog_metadata<"some_variadic_param", pos_or_kw, pos_vararg>,
  pog3 = #lit.pog_metadata<"some_constrained_param", pos_or_kw, not_vararg, {
    <1, loc("file.mojo":10:5)>
  }>,
  pog4 = #lit.pog_metadata<"some_very_constrained_param", pos_or_kw, pos_vararg, {
    <1, loc("file.mojo":10:5)>,
    <pog1, loc("file.mojo":11:5)>
  }>
} : () -> ()

// CHECK-LABEL: "pogs.with_defaults"
// CHECK-SAME: {pogs = #lit.pog_list<
// CHECK-SAME: [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
// CHECK-SAME: [4.200000e+00 : f32], [1 : i64]>}
"pogs.with_defaults"() {pogs = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  [4.2 : f32], [1: i64]
>} : () -> ()

// CHECK-LABEL: "pogs.with_variadics"
// CHECK-SAME: {pogs = #lit.pog_list<
// CHECK-SAME: [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, pack_vararg>],
// CHECK-SAME: [], [], owned_in_mem>}
"pogs.with_variadics"() {pogs = #lit.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, pack_vararg>],
  [], [], owned_in_mem
>} : () -> ()

// CHECK-LABEL: "empty.pogs"
// CHECK-SAME: {pogs = #lit.pog_list<[], [], []>}
"empty.pogs"() {pogs = #lit.pog_list<[], [], []>} : () -> ()

// CHECK-LABEL: "empty.metadata"
// CHECK-SAME: #lit.fn_metadata<<[], [], []>, 0>
"empty.metadata"() {metadata = #lit.fn_metadata<<[], [], []>, 0>} : () -> ()

// CHECK-LABEL: "some.metadata1"
// CHECK-SAME: #lit.fn_metadata
// CHECK-SAME: <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
// CHECK-SAME: 2, {mut lt}>
"some.metadata1"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, {mut lt}
>} : () -> ()

// CHECK-LABEL: "some.metadata2"
// CHECK-SAME: 2, {mut lt}, true>
"some.metadata2"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, {mut lt}, true
>} : () -> ()

// CHECK-LABEL: "some.metadata3"
// CHECK-SAME: 2, true>
"some.metadata3"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, true
>} : () -> ()

// CHECK-LABEL: "some.metadata4"
// CHECK-SAME: 2, false>
"some.metadata4"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, false
>} : () -> ()

// CHECK-LABEL: "some.metadata5"
// CHECK-SAME: 2 where {<1, #[[LOC1]]>, <0, #[[LOC2]]>}
"some.metadata5"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2 where {<1, #loc1>, <0, #loc2>}
>} : () -> ()

// CHECK-LABEL: "some.metadata6"
// CHECK-SAME: 2, false where {<1, #[[LOC1]]>, <0, #[[LOC2]]>}
"some.metadata6"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, false where {<1, #loc1>, <0, #loc2>}
>} : () -> ()

// CHECK-LABEL: "some.metadata7"
// CHECK-SAME: 2, {mut lt} where {<1, #[[LOC1]]>, <0, #[[LOC2]]>}
"some.metadata7"() {metadata = #lit.fn_metadata<
  <[<"someRef", pos, not_vararg>, <"v", kw, not_vararg>], [13 : index], [17 : i64]>,
  2, {mut lt} where {<1, #loc1>, <0, #loc2>}
>} : () -> ()

// CHECK-LABEL: "none.type"
// CHECK-SAME: #kgen.none : !kgen.none
"none.type"() {a = #kgen.none : !kgen.none} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK-LABEL: "struct.attr"
// CHECK-SAME: #lit.struct<{foo = 5, bar: dtype = f32}>
"struct.attr"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !lit.struct<@Foo>} : () -> ()

// CHECK-LABEL: "origin.attr"
// CHECK: #lit.any.origin : !lit.origin<1>
"origin.attr"() {a = #lit.any.origin : !lit.origin<1>} : () -> ()


kgen.generator @lifetime_lower<p: !lit.origin<1>>(%a: !lit.origin<0>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK: %origin = kgen.param.constant: origin<0> = <#lit.any.origin>
  %cst = kgen.param.constant: origin<0> = <#lit.any.origin>
  // CHECK: kgen.call @lifetime_lower<:origin<1> #lit.any.origin>(%origin) : (!lit.origin<0>) -> ()
  kgen.call @lifetime_lower<:origin<1> #lit.any.origin>(%cst) : (!lit.origin<0>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: origin<0>, q: origin<1>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm p>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, mut q>)
kgen.generator @ref_type<p: !lit.origin<0>, q: !lit.origin<1>>
(%a: !lit.ref<!lit.struct<@Foo>, imm p>, %b: !lit.ref<!lit.struct<@Foo>, mut q>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @field_attr<life: origin<0>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm life->field>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, imm life->a->b->c>)
kgen.generator @field_attr<life: !lit.origin<0>>
(%arg0: !lit.ref<@Foo, imm life->field>,
 %arg1: !lit.ref<@Foo, imm life->a->b->c>) {

  // CHECK-NEXT: "verbose_attr"() {attr = #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<0>, "field0"> : !lit.origin<0>} : () -> ()
  "verbose_attr"() {attr = #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<0>, "field0"> : !lit.origin<0>} : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @indirect_origin_attr<orig: origin<0>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm orig[]->field>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, imm orig[]->a[]>)
kgen.generator @indirect_origin_attr<orig: !lit.origin<0>>
(%arg0: !lit.ref<@Foo, imm orig[]->field>,
 %arg1: !lit.ref<@Foo, imm orig[]->a[]>) {

  // CHECK-NEXT: "verbose_attr"() {attr = #lit.indirect.origin<#kgen.param.decl.ref<"orig"> : !lit.origin<0>> : !lit.origin<0>} : () -> ()
  "verbose_attr"() {attr = #lit.indirect.origin<#kgen.param.decl.ref<"orig"> : !lit.origin<0>> : !lit.origin<0>} : () -> ()
  kgen.return
}




// CHECK-LABEL: kgen.generator @anystruct
kgen.generator @anystruct() {
  // CHECK: T: meta<!lit.struct<@Foo>> = <@Foo>
  kgen.param.declare T: meta<!lit.struct<@Foo>> = <@Foo>
  kgen.return
}

// CHECK-LABEL: kgen.generator @unpacked
kgen.generator @unpacked<a: variadic<index>>() {
  // CHECK-NEXT: constant = <#lit.unpacked<:variadic<index> a, kw>>
  %c = kgen.param.constant = <#lit.unpacked<:variadic<index> a, kw>>
  kgen.return
}

// CHECK-LABEL: @lifetime_union
kgen.generator @lifetime_union<x: !lit.origin<0>, y: !lit.origin<0>>() {
  // CHECK-NEXT: %a = lit.var.decl
  %a = lit.var.decl "a" imp : !lit.ref<index, mut z>

  // CHECK-NEXT: "a"() {a = #lit.any.origin : !lit.origin<0>} : () -> ()
  "a"() {a = #lit.origin.union<#lit.any.origin : !lit.origin<0>> : !lit.origin<0>} : () -> ()
  // CHECK-NEXT: "b"() {a = #kgen.param.decl.ref<"x"> : !lit.origin<0>}
  "b"() {a = #lit.origin.union<#kgen.param.decl.ref<"x"> :!lit.origin<0>>
        : !lit.origin<0>} : () -> ()
  // CHECK-NEXT: "c"() {a = #lit.origin.union<#kgen.param.decl.ref<"x"> : !lit.origin<0>, #kgen.param.decl.ref<"y"> : !lit.origin<0>> : !lit.origin<0>}
  "c"() {a = #lit.origin.union<#kgen.param.decl.ref<"x"> :!lit.origin<0>,
                                 #kgen.param.decl.ref<"y"> :!lit.origin<0>>
        : !lit.origin<0>} : () -> ()
  // CHECK-NEXT: "d"() {a = #lit.origin.union<
  // CHECK-SAME:          #kgen.param.decl.ref<"x"> : !lit.origin<0>,
  // CHECK-SAME:          #lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<0>, "field0"> : !lit.origin<0>
  // CHECK-SAME:        > : !lit.origin<0>}
  "d"() {a = #lit.origin.union<#kgen.param.decl.ref<"x"> :!lit.origin<0>,
                                 #lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<0>, "field0"> :!lit.origin<0>,
                                 #kgen.param.decl.ref<"x"> :!lit.origin<0>>
        : !lit.origin<0>} : () -> ()

  // CHECK-NEXT: "e"() {a = #lit.origin.union<#lit.origin.ref<0, 1> : !lit.origin<0>, #lit.origin.ref<1, 0> : !lit.origin<0>> : !lit.origin<0>}
  "e"() {a = #lit.origin.union<#lit.origin.ref<1, 0> : !lit.origin<0>, #lit.origin.ref<0, 1> : !lit.origin<0>> : !lit.origin<0>} : () -> ()

  kgen.param.declare is_mut: i1 = <0>
  kgen.param.declare a: origin<1> = <?>
  kgen.param.declare b: origin<0> = <?>
  kgen.param.declare c: origin<is_mut> = <?>

  // CHECK: <{(is_mut) c, mut a, imm b}>
  kgen.param.constant: origin.set = <{imm b, mut a, (is_mut) c, mut a}>
  // CHECK-NEXT: <{mut a}>
  kgen.param.constant: origin.set = <{imm (mutcast mut a)}>
  // CHECK-NEXT: <{mut a, mut #lit.any.origin, imm b}>
  kgen.param.constant: origin.set = <{mut #lit.any.origin, mut a, imm b}>
  // CHECK-NEXT: <{mut a, imm b}>
  kgen.param.constant: origin.set = <{mut {(mutcast imm b), a}}>

  // CHECK-NEXT: <{(mutcast mut a), b}>
  kgen.param.constant: origin<0> = <|{mut a, imm b}|>

  // CHECK-NEXT: kgen.param.declare nothing: origin<0> = <#lit.any.origin>
  kgen.param.declare nothing: !lit.origin<0> = <#lit.any.origin>
  // CHECK-NEXT:  kgen.param.declare nothing_2: origin<0> = <#lit.any.origin>
  kgen.param.declare nothing_2: !lit.origin<0> = <{#lit.any.origin, #lit.any.origin}>
  // CHECK-NEXT: kgen.param.declare x_ref: origin<0> = <x>
  kgen.param.declare x_ref: !lit.origin<0> = <x>
  // CHECK-NEXT: kgen.param.declare x_ref2: origin<0> = <x>
  kgen.param.declare x_ref2: !lit.origin<0> = <*"x">
  // CHECK-NEXT: kgen.param.declare x_or_y_ref: origin<0> = <{x, y}>
  kgen.param.declare x_or_y_ref: !lit.origin<0> = <{x, y, x}>
  // CHECK-NEXT: kgen.param.declare y_ref: origin<0> = <#lit.any.origin>
  kgen.param.declare y_ref: !lit.origin<0> = <{y, #lit.any.origin}>
  // CHECK-NEXT: kgen.param.declare xyz_ref: origin<0> = <{x, y, (mutcast mut z)}>
  kgen.param.declare xyz_ref: !lit.origin<0> = <{{x, y}, {(mutcast mut z), y}}>

  kgen.return
}

// CHECK-LABEL: kgen.generator @test_origin
kgen.generator @test_origin<x: i1>() {
  // CHECK-NEXT: kgen.param.constant: origin<0> = <rebind(:i1 x)>
  kgen.param.constant: origin<0> = <rebind(:i1 x)>
  kgen.return
}
