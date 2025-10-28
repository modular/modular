// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

// CHECK-DAG: #[[LOC_C1:.+]] = loc("test.mojo":10:5)
// CHECK-DAG: #[[LOC_C2:.+]] = loc("test.mojo":15:10)
// CHECK-DAG: #[[LOC_C3:.+]] = loc("test.mojo":20:15)

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

// CHECK: #kgen.type<index> : !kgen.type
"some.op"() {type = #kgen.type<index> : !kgen.type} : () -> ()
// CHECK: a = #kgen.type<array<1, i1>> : !kgen.type
// CHECK: b = #kgen.type<array<apply(:() -> index @return_one), i1>> : !kgen.type
// CHECK: c = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type
// CHECK: d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type
// CHECK: e = #kgen.type<array<1, i1>> : !kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>> : !kgen.type,
  c = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type,
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
  // CHECK-SAME: b = #kgen.get_witness<#kgen.genref<@LinkedList<:type index, 3>>, "Boolable", "__bool__"> : !kgen.generator<(!kgen.struct<(index, pointer<none>)>) -> i1>,
  b = #kgen.get_witness<#kgen.genref<@LinkedList<:type index, 3>>, "Boolable", "__bool__"> : !kgen.generator<(!kgen.struct<(index, pointer<none>)>) -> i1>,
  // CHECK-SAME: c = #kgen.get_linkage_name<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string,
  c = #kgen.get_linkage_name<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string,
  // CHECK-SAME: d = #kgen.get_type_name<#kgen.genref<@LinkedList<:type index, 3>>, true> : !kgen.string,
  d = #kgen.get_type_name<#kgen.genref<@LinkedList<:type index, 3>>, true> : !kgen.string,
  // CHECK-SAME: e = #kgen.compile_offload_closure<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string
  e = #kgen.compile_offload_closure<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string,
  // CHECK-SAME: f = #kgen.compile_assembly<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, =llvm, "", false, :() -> index @return_one> : !kgen.string,
  f = #kgen.compile_assembly<#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>, =llvm, "", false, :() -> index @return_one> : !kgen.string,
  // CHECK-SAME: g = #kgen.get_source_name<#kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string
  g = #kgen.get_source_name<#kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>> : !kgen.string
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

"some.op"() {
  // COM: downcast is folded in evaluation context, not by AttrBuilder
  // CHECK: identityDowncast = #kgen.downcast<array<1, i1>> : !kgen.type,
  identityDowncast = #kgen.downcast<#kgen.type<array<1, i1>> : !kgen.type> : !kgen.type,
  // CHECK-SAME: identityUpcast = #kgen.type<array<1, i1>> : !kgen.type
  identityUpcast = #kgen.upcast<#kgen.type<array<1, i1>> : !kgen.type> : !kgen.type
} : () -> ()

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
  // CHECK-SAME: @bar_move<:type index, :type index>, @bar_del<:type index, :type index> move : !kgen.pointer<struct<(index, index)>>>
  b = #kgen.mem_symbol_triple<@bar_copy<:type index, :type index>,
                              @bar_move<:type index, :type index>,
                              @bar_del<:type index, :type index> move : !kgen.pointer<struct<(index, index)>>>
} : () -> ()

"some.op"() {
  a = 5 : index,
  // CHECK: gen0 = #kgen.gen<add(a, 3)> : !kgen.generator<<>index>
  gen0 = #kgen.gen<add(a, 3)> : !kgen.generator<<>index>,
  // CHECK-SAME: gen1 = #kgen.gen<add(*(0,0), 1)> : !kgen.generator<<index>index>
  gen1 = #kgen.gen<add(*(0,0), 1)> : !kgen.generator<<index> index>,
  // CHECK-SAME: gen2 = #kgen.gen<add(*(0,0), *(0,1))> : !kgen.generator<<index, index>index>
  gen2 = #kgen.gen<add(*(0,0), *(0,1))> : !kgen.generator<<index, index> index>
} : () -> ()

"some.op"() {
  a = 5 : index,
  // CHECK: constraint1 = #kgen.constraint<1, #[[LOC_C1]]>
  constraint1 = #kgen.constraint<1, loc("test.mojo":10:5)>,
  // CHECK-SAME: constraint2 = #kgen.constraint<ge(a, 4), #[[LOC_C2]]>
  constraint2 = #kgen.constraint<ge(a, 4), loc("test.mojo":15:10)>,
  // CHECK-SAME: #kgen.constraint<conforms_to(:type array<1, i1>, ["trait_1", "trait_2"]), #[[LOC_C3]]>
  constraint3 = #kgen.constraint<conforms_to(:type array<1, i1>, ["trait_1", "trait_2"]), loc("test.mojo":20:15)>
} : () -> ()

// CHECK: llvm_bitcode_lib_unused = #kgen.llvm.bitcode.lib<used = false, library = "/path/to/lib.bc">
// CHECK-SAME: llvm_bitcode_lib_used = #kgen.llvm.bitcode.lib<used = true, library = "/opt/libs/math.bc">
// CHECK-SAME: llvm_bitcode_libs = #kgen<llvm.bitcode.libs[<used = false, library = "/path/to/lib1.bc">, <used = true, library = "/path/to/lib2.bc">]>
// CHECK-SAME: llvm_bitcode_libs_empty = #kgen<llvm.bitcode.libs[]>
"some.op"() {
  llvm_bitcode_lib_unused = #kgen<llvm.bitcode.lib<used = false, library = "/path/to/lib.bc">>,
  llvm_bitcode_lib_used = #kgen<llvm.bitcode.lib<used = true, library = "/opt/libs/math.bc">>,
  llvm_bitcode_libs = #kgen<llvm.bitcode.libs[
    #kgen<llvm.bitcode.lib<used = false, library = "/path/to/lib1.bc">>,
    #kgen<llvm.bitcode.lib<used = true, library = "/path/to/lib2.bc">>
  ]>,
  llvm_bitcode_libs_empty = #kgen<llvm.bitcode.libs[]>
} : () -> ()

"some.op"() {
    // CHECK: a = 139 : ui8,
    a = #pop.dtype_to_ui8<si32> : ui8,
    // CHECK-SAME: b = #pop.dtype_to_ui8<foo> : ui8
    b = #pop.dtype_to_ui8<foo>  : ui8
} : () -> ()

"some.op"() {
  // CHECK: a = #pop.simd<2, 5> : !pop.simd<2, si32>
  a = #pop.cast_from_builtin< #M.dense_array<2, 5> : vector<2xsi32>> : !pop.simd<2, si32>,
  // CHECK: b = #pop<simd "0"> : !pop.scalar<f8e5m2>
  b = #pop.cast_from_builtin< 0.0 : f8E5M2> : !pop.scalar<f8e5m2>,
  // CHECK: c = #pop<simd "0"> : !pop.scalar<f8e5m2fnuz>
  c = #pop.cast_from_builtin< 0.0 : f8E5M2FNUZ> : !pop.scalar<f8e5m2fnuz>,
  // CHECK: d = #pop<simd "0"> : !pop.scalar<f8e4m3fn>
  d = #pop.cast_from_builtin< 0.0 : f8E4M3FN> : !pop.scalar<f8e4m3fn>,
  // CHECK: e = #pop<simd "0"> : !pop.scalar<f8e4m3fnuz>
  e = #pop.cast_from_builtin< 0.0 : f8E4M3FNUZ> : !pop.scalar<f8e4m3fnuz>,
  // CHECK: f = #pop<simd "0"> : !pop.scalar<f8e3m4>
  f = #pop.cast_from_builtin< 0.0 : f8E3M4> : !pop.scalar<f8e3m4>,
  // CHECK: g = #pop<simd "0"> : !pop.scalar<bf16>
  g = #pop.cast_from_builtin< 0.0 : bf16> : !pop.scalar<bf16>,
  // CHECK: h = #pop<simd -1> : !pop.scalar<si64>
  h = #pop.cast_from_builtin< -1 : si64> : !pop.scalar<si64>,
  // CHECK: i = #pop<simd 18446744073709551615> : !pop.scalar<ui64>
  i = #pop.cast_from_builtin< 0xffffffffffffffff : ui64> : !pop.scalar<ui64>,
  // CHECK: j = #pop<simd 0> : !pop.scalar<si128>
  j = #pop.cast_from_builtin< 0 : si128> : !pop.scalar<si128>
} : () -> ()
