// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

// CHECK: *"mangled_fn{{.*}}int
"some.op"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!lit.struct<_\22int\22::_Int>])" : index>} : () -> ()

kgen.generator @return_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: a = #kgen.type
// CHECK-SAME: b = #kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>> : !kgen.type
} : () -> ()

// CHECK: #kgen.param.index.ref<0, 0> : index
"some.op"() {ref = #kgen.param.index.ref<0, 0> : index} : () -> ()

// CHECK: #pop.int_literal<5> : !pop.int_literal
"some.op"() {data = #pop.int_literal<5> : !pop.int_literal} : () -> ()

// CHECK: #pop.float_literal<5|3> : !pop.float_literal
"some.op"() {data = #pop.float_literal<5|3> : !pop.float_literal} : () -> ()
// CHECK: #pop.float_literal<neg_zero> : !pop.float_literal
"some.op"() {data = #pop.float_literal<neg_zero> : !pop.float_literal} : () -> ()
// CHECK: #pop.float_literal<inf> : !pop.float_literal
"some.op"() {data = #pop.float_literal<inf> : !pop.float_literal} : () -> ()

// CHECK: #kgen.env<{bar = 1 : index, foo}>
"some.op"() {env = #kgen.env<{bar = 1 : index, foo}>} : () -> ()

// CHECK: #kgen<decorators[1 : i64]>
"some.op"() {decorators = #kgen<decorators[1 : i64]>} : () -> ()

// CHECK: #pop.int_literal<1234>
// CHECK-SAME: #pop.int_literal<12345678901234567899012345678901234567890>
"some.op"() {a = #pop.int_literal<1234> : !pop.int_literal,
             b = #pop.int_literal<12345678901234567899012345678901234567890> : !pop.int_literal} : () -> ()

// CHECK-LABEL: @struct_constants
kgen.generator @struct_constants<T: type, A: !kgen.param<T>, value: !pop.scalar<f32>>() {
  // CHECK: struct<(index, f32)> = <{ 1, 2.5{{0+}}e+00 }>
  kgen.param.constant: struct<(index, f32)> = <{ 1, 2.5 }>
  // CHECK: struct<(scalar<f32>)> = <{ value }>
  kgen.param.constant: struct<(scalar<f32>)> = <{ value }>
  // CHECK: struct<(T)> = <{ A }>
  kgen.param.constant: struct<(T)> = <{ A }>
  kgen.return
}

// CHECK-LABEL: @pack_constants
kgen.generator @pack_constants<Ts: variadic<i32>>() {
  // CHECK: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  %0 = kgen.param.constant: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  // CHECK: !kgen.pack<[]> = <<>>
  %1 = kgen.param.constant: !kgen.pack<[]> = <<>>
  kgen.return
}


// CHECK-LABEL: @variant_constants
kgen.generator @variant_constants<T: type, U: type, value: !kgen.param<T>>() {
  // CHECK: variant<f32, f64> = <{:f32 2.5{{0+}}e+00, 0}>
  %0 = kgen.param.constant: variant<f32, f64> = <{:f32 2.5, 0}>
  // CHECK: variant<T, U> = <{:!kgen.param<T> value, 0}>
  %1 = kgen.param.constant: variant<T, U> = <{:!kgen.param<T> value, 0}>
  kgen.return
}

kgen.generator @entry1() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}
kgen.generator @entry2() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: vtableEntry = #kgen<vtable.entry"entry1" : () -> index = @entry1>
"some.op"() {vtableEntry = #kgen<vtable.entry "entry1" : () -> index = @entry1>} : () -> ()
// CHECK: vtable = #kgen<vtable"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2
"some.op"() {vtable = #kgen<vtable "entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2>} : () -> ()

// CHECK: #kgen.type<index, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
"some.op"() {type = #kgen.type<index, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type} : () -> ()
// CHECK: a = #kgen.type<array<1, i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: b = #kgen.type<array<apply(:() -> index @return_one), i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: c = #kgen.type<array<1, i1>, array<2, i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type
// CHECK: e = #kgen.type<array<1, i1>> : !kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  c = #kgen.type<array<1, i1>, array<2, i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type,
  e = #kgen.type<array<1, i1>, array<1, i1>> : !kgen.type
} : () -> ()

// CHECK: #kgen<tailkind none>
// CHECK: #kgen<tailkind musttail>
// CHECK: #kgen<tailkind notail>
"some.op"() {
  a = #kgen<tailkind none>,
  c = #kgen<tailkind musttail>,
  d = #kgen<tailkind notail>
} : () -> ()

// CHECK: kgen.struct.generator @LinkedList<T: type, x: !kgen.param<T>> = struct_inst<
// CHECK-SAME:   "LinkedList"
// CHECK-SAME:   [T, x]
// CHECK-SAME:   <:type T, :!kgen.param<T> x>
// CHECK-SAME:   (data: typevalue<T>,
// CHECK-SAME:    next: typevalue<#kgen.genref<@LinkedList<:type T, :!kgen.param<T> x>>>)
kgen.struct.generator @LinkedList<T: type, x: !kgen.param<T>> =
  struct_inst<"LinkedList"[T, x]<:type T, :!kgen.param<T> x>(
    data: typevalue<T>,
    next: typevalue<#kgen.genref<@LinkedList<:type T, :!kgen.param<T> x>>>
  )>
{
  kgen.conformance @Boolable {
    kgen.witness "__bool__" : (!kgen.struct<(T, pointer<none>)>) -> i1 = @"LinkedList::__bool__(::LinkedList)"<:type T, :!kgen.param<T> x>
  }
}

kgen.generator @"LinkedList::__bool__(::LinkedList)"<T: type, x: !kgen.param<T>>(%arg0: !kgen.struct<(T, pointer<none>)>) -> i1 {
  %index1 = kgen.param.constant : i1 = <1>
  kgen.return %index1 : i1
}


"some.op"() {
  // CHECK: a = #kgen.genref<@LinkedList<:type index, 3>>
  a = #kgen.genref<@LinkedList<:type index, 3>>,
  // CHECK-SAME: b = #kgen.get_witness<#kgen.genref<@LinkedList<:type index, 3>>, "Boolable", "__bool__"> : !kgen.generator<(!kgen.struct<(index, pointer<none>)>) -> i1>
  b = #kgen.get_witness<#kgen.genref<@LinkedList<:type index, 3>>, "Boolable", "__bool__"> : !kgen.generator<(!kgen.struct<(index, pointer<none>)>) -> i1>
} : () -> ()


// CHECK: kgen.param.assert <rebind(:i53 42)>, "rebind must fold"
kgen.param.assert <rebind(:i73 rebind(:i53 42))>, "rebind must fold"

// COM: Closure Attribute

// CHECK-LABEL: kgen.generator export @bindIt() {
kgen.generator export @bindIt(){
  // CHECK-NEXT: kgen.param.declare a: !kgen.param_closure<@foo "fn"> = <#kgen.closure<@foo "fn">>
  kgen.param.declare a : !kgen.param_closure<@foo "fn"> = <#kgen.closure<@foo "fn">>
  kgen.return
}

// CHECK-LABEL: kgen.generator @closureSymbol()
kgen.generator @closureSymbol(){
  // CHECK: kgen.param.declare symbol: <!kgen.param_closure<@foo "fn">>
  // CHECK-SAME: (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index =
  // CHECK-SAME: <#kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn"> ?>>>
  kgen.param.declare symbol : <!kgen.param_closure<@foo "fn">>(!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index =
    <#kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn"> ?> >>
  kgen.return
}

"some.op"() {
  // CHECK: a = #kgen.mem_symbol_triple<@bar_move<:type index, :type index>,
  // CHECK-SAME: @bar_del<:type index, :type index> : !kgen.pointer<struct<(index, index)>>>
  a = #kgen.mem_symbol_triple<@bar_move<:type index, :type index>,
                              @bar_del<:type index, :type index> : !kgen.pointer<struct<(index, index)>>>,
  // CHECK: b = #kgen.mem_symbol_triple<@bar_copy<:type index, :type index>,
  // CHECK-SAME: @bar_move<:type index, :type index>, @bar_del<:type index, :type index> : !kgen.pointer<struct<(index, index)>>>
  b = #kgen.mem_symbol_triple<@bar_copy<:type index, :type index>,
                              @bar_move<:type index, :type index>,
                              @bar_del<:type index, :type index> : !kgen.pointer<struct<(index, index)>>>
} : () -> ()
