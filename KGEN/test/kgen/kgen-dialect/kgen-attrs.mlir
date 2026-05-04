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
kgen.generator @struct_constants<T: type, A: !kgen.param<T>, value: !kgen.scalar<f32>>() {
  // CHECK: struct<(index, f32)> = <{ 1, 2.5{{0+}}e+00 }>
  kgen.param.constant: struct<(index, f32)> = <{ 1, 2.5 }>
  // CHECK: struct<(scalar<f32>)> = <{ value }>
  kgen.param.constant: struct<(scalar<f32>)> = <{ value }>
  // CHECK: struct<(T)> = <{ A }>
  kgen.param.constant: struct<(T)> = <{ A }>
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
// CHECK: f = #kgen.param.decl.ref<"a"> : !kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>> : !kgen.type,
  c = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type,
  d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type,
  e = #kgen.type<array<1, i1>, array<1, i1>> : !kgen.type,
  f = #kgen.type<typevalue<a>, !kgen.param<a>> : !kgen.type
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
// CHECK-SAME:   (data: T,
// CHECK-SAME:    next: [typevalue<#kgen.genref<@LinkedList<:type T, :!kgen.param<T> x>>>, pointer<none>])
kgen.struct.generator @LinkedList<T: type, x: !kgen.param<T>> =
  struct_inst<"LinkedList"[T, x]<:type T, :!kgen.param<T> x>(
    data: [typevalue<T>, !kgen.param<T>],
    next: [typevalue<#kgen.genref<@LinkedList<:type T, :!kgen.param<T> x>>>, pointer<none>]
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

!BaseTrait = !lit.trait<@BaseTrait>
!DerivedTrait = !lit.trait<@DerivedTrait>

// CHECK-LABEL: kgen.generator @canonicalize_get_witness_upcast
kgen.generator @canonicalize_get_witness_upcast<T: !DerivedTrait>() {
  // CHECK-NOT: upcast
  // CHECK: #kgen.get_witness<:trait<@BaseTrait> T, "BaseTrait", "AssociatedType">
  kgen.param.constant: !kgen.type = <#kgen.get_witness<upcast(:!BaseTrait T), "BaseTrait", "AssociatedType"> : !kgen.type>
  kgen.return
}

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
  meta = #kgen.type<!kgen.type> : !kgen.type,
  // COM: non-trivial downcast is folded in evaluation context, not by AttrBuilder
  // CHECK: constantDowncast = #kgen.downcast<array<1, i1>> : !kgen.param<*"meta">,
  constantDowncast = #kgen.downcast<#kgen.type<array<1, i1>> : !kgen.type> : !kgen.param<*"meta">,
  // CHECK: identityDowncast = #kgen.type<array<1, i1>> : !kgen.type,
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

kgen.struct.generator @WitnessedMemStruct = struct_inst<"WitnessedMemStruct"(data: index)> {
  kgen.conformance @MemOps {
    kgen.witness "copy" : (!kgen.pointer<struct<(index)>> read_mem,
                           !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none = @witnessed_mem_copy
    kgen.witness "move" : (!kgen.pointer<struct<(index)>> owned_in_mem,
                           !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none = @witnessed_mem_move
    kgen.witness "del" : (!kgen.pointer<struct<(index)>> owned_in_mem) -> !kgen.none = @witnessed_mem_del
  }
}

kgen.generator @witnessed_mem_copy(%arg0: !kgen.pointer<struct<(index)>> read_mem,
                                   %arg1: !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @witnessed_mem_move(%arg0: !kgen.pointer<struct<(index)>> owned_in_mem,
                                   %arg1: !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @witnessed_mem_del(%arg0: !kgen.pointer<struct<(index)>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// COM: GetWitness-backed mem_symbol_triple should round-trip too.
"witness.mem.symbol.triple"() {
  // CHECK: c = #kgen.mem_symbol_triple<#kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "copy"> : !kgen.generator<(!kgen.pointer<struct<(index)>> read_mem, !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none>,
  // CHECK-SAME: #kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "move"> : !kgen.generator<(!kgen.pointer<struct<(index)>> owned_in_mem, !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none>,
  // CHECK-SAME: #kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "del"> : !kgen.generator<(!kgen.pointer<struct<(index)>> owned_in_mem) -> !kgen.none> move>
  c = #kgen.mem_symbol_triple<
        #kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "copy"> : !kgen.generator<(!kgen.pointer<struct<(index)>> read_mem, !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none>,
        #kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "move"> : !kgen.generator<(!kgen.pointer<struct<(index)>> owned_in_mem, !kgen.pointer<struct<(index)>> byref_result) -> !kgen.none>,
        #kgen.get_witness<#kgen.genref<@WitnessedMemStruct>, "MemOps", "del"> : !kgen.generator<(!kgen.pointer<struct<(index)>> owned_in_mem) -> !kgen.none> move>
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
  // CHECK-SAME: #kgen.constraint<conforms_to(:type array<1, i1>, [@trait_1, @trait_2]), #[[LOC_C3]]>
  constraint3 = #kgen.constraint<conforms_to(:type array<1, i1>, [@trait_1, @trait_2]), loc("test.mojo":20:15)>
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
  // CHECK: a = #kgen.simd<2, 5> : !kgen.simd<2, si32>
  a = #kgen.cast_from_builtin< #M.dense_array<2, 5> : vector<2xsi32>> : !kgen.simd<2, si32>,
  // CHECK: b = #kgen<simd "0"> : !kgen.scalar<f8e5m2>
  b = #kgen.cast_from_builtin< 0.0 : f8E5M2> : !kgen.scalar<f8e5m2>,
  // CHECK: c = #kgen<simd "0"> : !kgen.scalar<f8e5m2fnuz>
  c = #kgen.cast_from_builtin< 0.0 : f8E5M2FNUZ> : !kgen.scalar<f8e5m2fnuz>,
  // CHECK: d = #kgen<simd "0"> : !kgen.scalar<f8e4m3fn>
  d = #kgen.cast_from_builtin< 0.0 : f8E4M3FN> : !kgen.scalar<f8e4m3fn>,
  // CHECK: e = #kgen<simd "0"> : !kgen.scalar<f8e4m3fnuz>
  e = #kgen.cast_from_builtin< 0.0 : f8E4M3FNUZ> : !kgen.scalar<f8e4m3fnuz>,
  // CHECK: f = #kgen<simd "0"> : !kgen.scalar<f8e3m4>
  f = #kgen.cast_from_builtin< 0.0 : f8E3M4> : !kgen.scalar<f8e3m4>,
  // CHECK: g = #kgen<simd "0"> : !kgen.scalar<bf16>
  g = #kgen.cast_from_builtin< 0.0 : bf16> : !kgen.scalar<bf16>,
  // CHECK: h = #kgen<simd -1> : !kgen.scalar<si64>
  h = #kgen.cast_from_builtin< -1 : si64> : !kgen.scalar<si64>,
  // CHECK: i = #kgen<simd 18446744073709551615> : !kgen.scalar<ui64>
  i = #kgen.cast_from_builtin< 0xffffffffffffffff : ui64> : !kgen.scalar<ui64>,
  // CHECK: j = #kgen<simd 0> : !kgen.scalar<si128>
  j = #kgen.cast_from_builtin< 0 : si128> : !kgen.scalar<si128>
} : () -> ()

"some.op"() {
    // CHECK: a = #kgen.dtype.constant<si32>
    a = #pop.dtype_from_ui8<139 : ui8> : !dtype.si32,
    // CHECK: b = #kgen.dtype.constant<si16>
    b = #pop.dtype_from_ui8<137 : ui8> : !dtype.si16,
    // CHECK: c = #kgen.dtype.constant<f8e5m2>
    c = #pop.dtype_from_ui8<77 : ui8> : !dtype.f8e5m2
} : () -> ()

"some.op"() {
  // CHECK: a = 1 : si32
  a = #kgen.cast_to_builtin< #kgen<simd 1> : !kgen.simd<1, si32>> : si32,
  // CHECK: b = #M.dense_array<1, 1> : vector<2xsi32>,
  b = #kgen.cast_to_builtin< #kgen.simd<1, 1> : !kgen.simd<2, si32>> : vector<2xsi32>,
  // CHECK: c = #M.dense_array<1, 2> : vector<2xsi32>,
  c = #kgen.cast_to_builtin< #kgen.simd<1, 2> : !kgen.simd<2, si32>> : vector<2xsi32>,
  // CHECK: d = 1.000000e+00 : f16
  d = #kgen.cast_to_builtin< #kgen<simd "1.0"> : !kgen.simd<1, f16>> : f16
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 1> : !kgen.scalar<ui32>
  a = #pop.cast< #kgen<simd 1> : !kgen.simd<1, si32>> : !kgen.simd<1, ui32>,
  // CHECK: b = #kgen<simd 4294967295> : !kgen.scalar<ui32>
  b = #pop.cast< #kgen<simd -1> : !kgen.simd<1, si32>> : !kgen.simd<1, ui32>,
  // CHECK: c = #kgen.simd<65534, 65535, 0, 1>
  c = #pop.cast< #kgen.simd<-2, -1, 0, 1> : !kgen.simd<4, si8>> : !kgen.simd<4, ui16>,
  // CHECK: d = #kgen.simd<"2.5", "1.29980469", "0">
  d = #pop.cast< #kgen.simd<"2.5", "1.3", "0.0"> : !kgen.simd<3, f16>> : !kgen.simd<3, f32>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 1> : !kgen.simd<4, si32>
  a = #pop.simd_splat< #kgen<simd 1> : !kgen.scalar<si32>> : !kgen.simd<4, si32>,
  // CHECK: b = #kgen<simd "1"> : !kgen.simd<3, f16>
  b = #pop.simd_splat< #kgen<simd "1.0"> : !kgen.scalar<f16>> : !kgen.simd<3, f16>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 0> : !kgen.scalar<si32>
  a = #pop.simd_and< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<2, 42, 1024, 0>
  b = #pop.simd_and< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_and<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_and< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 3> : !kgen.scalar<si32>
  a = #pop.simd_xor< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<5, -43, -1025, -1>
  b = #pop.simd_xor< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_xor<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_xor< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 3> : !kgen.scalar<si32>
  a = #pop.simd_or< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<7, -1, -1, -1>
  b = #pop.simd_or< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_or<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_or< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 3> : !kgen.scalar<si32>
  a = #pop.simd_add< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<9, 41, 1023, -1>
  b = #pop.simd_add< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_add<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_add< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>,

  // CHECK: d = #kgen<simd false> : !kgen.scalar<bool>
  d = #pop.simd_add< #kgen<simd true> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: e = #kgen<simd true> : !kgen.scalar<bool>
  e = #pop.simd_add< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: f = #kgen<simd false> : !kgen.scalar<bool>
  f = #pop.simd_add< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd false> : !kgen.scalar<bool>> : !kgen.scalar<bool>,

  // CHECK: g = #kgen<simd "3.5"> : !kgen.scalar<f32>
  g = #pop.simd_add< #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "2.5"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,
  // CHECK: h = #kgen<simd "NaN"> : !kgen.scalar<f32>
  h = #pop.simd_add< #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,

  // CHECK: i = #kgen<simd 6> : !kgen.scalar<index>
  i = #pop.simd_add< #kgen<simd 2> : !kgen.scalar<index>, #kgen<simd 4> : !kgen.scalar<index>> : !kgen.scalar<index>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd -1> : !kgen.scalar<si32>
  a = #pop.simd_sub< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<5, 43, -1025, 1>
  b = #pop.simd_sub< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_sub<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_sub< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>,

  // CHECK: d = #kgen<simd false> : !kgen.scalar<bool>
  d = #pop.simd_sub< #kgen<simd true> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: e = #kgen<simd true> : !kgen.scalar<bool>
  e = #pop.simd_sub< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: f = #kgen<simd false> : !kgen.scalar<bool>
  f = #pop.simd_sub< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd false> : !kgen.scalar<bool>> : !kgen.scalar<bool>,

  // CHECK: g = #kgen<simd "-1.5"> : !kgen.scalar<f32>
  g = #pop.simd_sub< #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "2.5"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,
  // CHECK: h = #kgen<simd "NaN"> : !kgen.scalar<f32>
  h = #pop.simd_sub< #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,

  // CHECK: i = #kgen<simd -2> : !kgen.scalar<index>
  i = #pop.simd_sub< #kgen<simd 2> : !kgen.scalar<index>, #kgen<simd 4> : !kgen.scalar<index>> : !kgen.scalar<index>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 2> : !kgen.scalar<si32>
  a = #pop.simd_mul< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<14, -42, -1024, 0>
  b = #pop.simd_mul< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_mul<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_mul< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>,

  // CHECK: d = #kgen<simd true> : !kgen.scalar<bool>
  d = #pop.simd_mul< #kgen<simd true> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: e = #kgen<simd false> : !kgen.scalar<bool>
  e = #pop.simd_mul< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: f = #kgen<simd false> : !kgen.scalar<bool>
  f = #pop.simd_mul< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd false> : !kgen.scalar<bool>> : !kgen.scalar<bool>,

  // CHECK: g = #kgen<simd "5"> : !kgen.scalar<f32>
  g = #pop.simd_mul< #kgen<simd "2.0"> : !kgen.scalar<f32>, #kgen<simd "2.5"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,
  // CHECK: h = #kgen<simd "NaN"> : !kgen.scalar<f32>
  h = #pop.simd_mul< #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,

  // CHECK: i = #kgen<simd 8> : !kgen.scalar<index>
  i = #pop.simd_mul< #kgen<simd 2> : !kgen.scalar<index>, #kgen<simd 4> : !kgen.scalar<index>> : !kgen.scalar<index>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 3> : !kgen.scalar<si32>
  a = #pop.simd_div< #kgen<simd 6> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<3, -42, 0, 0>
  b = #pop.simd_div< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>,
                     #kgen.simd<2, -1, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_div<#kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_div< #kgen.unknown : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>>,

  // Integer division by zero should not fold (undefined behavior)
  // CHECK: d = #pop.simd_div<#kgen<simd 6> : !kgen.scalar<si32>, #kgen<simd 0> : !kgen.scalar<si32>> : !kgen.scalar<si32>
  d = #pop.simd_div< #kgen<simd 6> : !kgen.scalar<si32>, #kgen<simd 0> : !kgen.scalar<si32>>,

  // CHECK: e = #kgen<simd "2.5"> : !kgen.scalar<f32>
  e = #pop.simd_div< #kgen<simd "5.0"> : !kgen.scalar<f32>, #kgen<simd "2.0"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,

  // Float division by zero folds to inf (IEEE 754)
  // CHECK: f = #kgen<simd "+Inf"> : !kgen.scalar<f32>
  f = #pop.simd_div< #kgen<simd "5.0"> : !kgen.scalar<f32>, #kgen<simd "0.0"> : !kgen.scalar<f32>> : !kgen.scalar<f32>,

  // CHECK: g = #kgen<simd 2> : !kgen.scalar<index>
  g = #pop.simd_div< #kgen<simd 8> : !kgen.scalar<index>, #kgen<simd 4> : !kgen.scalar<index>> : !kgen.scalar<index>,

  // CHECK: h = #kgen<simd true> : !kgen.scalar<bool>
  h = #pop.simd_div< #kgen<simd true> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // CHECK: i = #kgen<simd false> : !kgen.scalar<bool>
  i = #pop.simd_div< #kgen<simd false> : !kgen.scalar<bool>, #kgen<simd true> : !kgen.scalar<bool>> : !kgen.scalar<bool>,
  // Bool division by zero (false) should not fold
  // CHECK: j = #pop.simd_div<#kgen<simd true> : !kgen.scalar<bool>, #kgen<simd false> : !kgen.scalar<bool>> : !kgen.scalar<bool>
  j = #pop.simd_div< #kgen<simd true> : !kgen.scalar<bool>, #kgen<simd false> : !kgen.scalar<bool>>
} : () -> ()

"some.op"() {
  // CHECK: a = #pop.simd_cmp<eq, #kgen.unknown : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<bool>
  a = #pop.simd_cmp<eq, #kgen.unknown : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<bool>,
  // CHECK: b = #kgen<simd false> : !kgen.scalar<bool>
  b = #pop.simd_cmp<eq, #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<bool>,
  // CHECK: c = #kgen<simd true> : !kgen.scalar<bool>
  c = #pop.simd_cmp<eq, #kgen<simd 42> : !kgen.scalar<si32>, #kgen<simd 42> : !kgen.scalar<si32>> : !kgen.scalar<bool>,
  // CHECK: d = #kgen<simd false> : !kgen.scalar<bool>
  d = #pop.simd_cmp<lt, #kgen<simd 42> : !kgen.scalar<si32>, #kgen<simd 42> : !kgen.scalar<si32>> : !kgen.scalar<bool>,
  // CHECK: e = #kgen<simd true> : !kgen.scalar<bool>
  e = #pop.simd_cmp<le, #kgen<simd 42> : !kgen.scalar<si32>, #kgen<simd 42> : !kgen.scalar<si32>> : !kgen.scalar<bool>,
  // CHECK: f = #kgen<simd true> : !kgen.scalar<bool>
  f = #pop.simd_cmp<eq, #kgen<sugar alias, !kgen.scalar<ui8>, *?, #kgen<simd 5>>, #kgen<simd 5> : !kgen.scalar<ui8>> : !kgen.scalar<bool>
} : () -> ()

"some.op"() {
  // CHECK: a0 = #kgen<simd true> : !kgen.scalar<bool>
  a0 = #pop.simd_cmp<eq, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: a1 = #kgen<simd false> : !kgen.scalar<bool>
  a1 = #pop.simd_cmp<eq, #kgen<simd "1.5"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: a2 = #kgen<simd false> : !kgen.scalar<bool>
  a2 = #pop.simd_cmp<eq, #kgen<simd "NaN"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: b0 = #kgen<simd false> : !kgen.scalar<bool>
  b0 = #pop.simd_cmp<lt, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: b1 = #kgen<simd true> : !kgen.scalar<bool>
  b1 = #pop.simd_cmp<lt, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "1.01"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: b2 = #kgen<simd false> : !kgen.scalar<bool>
  b2 = #pop.simd_cmp<lt, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: b3 = #kgen<simd false> : !kgen.scalar<bool>
  b3 = #pop.simd_cmp<lt, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: b4 = #kgen<simd false> : !kgen.scalar<bool>
  b4 = #pop.simd_cmp<lt, #kgen<simd "NaN"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: c0 = #kgen<simd true> : !kgen.scalar<bool>
  c0 = #pop.simd_cmp<le, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: c1 = #kgen<simd true> : !kgen.scalar<bool>
  c1 = #pop.simd_cmp<le, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "1.01"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: c2 = #kgen<simd false> : !kgen.scalar<bool>
  c2 = #pop.simd_cmp<le, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: c3 = #kgen<simd false> : !kgen.scalar<bool>
  c3 = #pop.simd_cmp<le, #kgen<simd "1.0"> : !kgen.scalar<f32>, #kgen<simd "NaN"> : !kgen.scalar<f32>> : !kgen.scalar<bool>,
  // CHECK: c4 = #kgen<simd false> : !kgen.scalar<bool>
  c4 = #pop.simd_cmp<le, #kgen<simd "NaN"> : !kgen.scalar<f32>, #kgen<simd "1.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>
} : () -> ()

// Index comparisons fold when 32-bit and 64-bit results agree.
// Comparisons that disagree remain unfolded (require target info).
"some.op"() {
  // CHECK: a = #kgen<simd true> : !kgen.scalar<bool>
  a = #pop.simd_cmp<eq, #kgen<simd 5> : !kgen.scalar<index>, #kgen<simd 5> : !kgen.scalar<index>> : !kgen.scalar<bool>,
  // CHECK: b = #pop.simd_cmp<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>
  b = #pop.simd_cmp<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd -1> : !kgen.scalar<si32>
  a = #pop.simd_neg< #kgen<simd 1> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<-7, -42, 1, 0>
  b = #pop.simd_neg< #kgen.simd<7, 42, -1, 0> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: c = #pop.simd_neg<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  c = #pop.simd_neg< #kgen.unknown : !kgen.simd<4, si32>>,
  // CHECK: d = #kgen.simd<false, true>
  d = #pop.simd_neg< #kgen.simd<true, false> : !kgen.simd<2, bool>> : !kgen.simd<2, bool>,
  // CHECK: e = #kgen.simd<"-1", "1.5", "-0", "0", "NaN", "NaN", "-Inf", "+Inf">
  e = #pop.simd_neg< #kgen.simd<"1.0", "-1.5", "0.0", "-0.0", "NaN", "-NaN", "inf", "-inf"> : !kgen.simd<8, f32>> : !kgen.simd<8, f32>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 7> : !kgen.scalar<si32>
  a = #pop.simd_floor< #kgen<simd 7> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<3, -4>
  b = #pop.simd_floor< #kgen.simd<3, -4> : !kgen.simd<2, si32>> : !kgen.simd<2, si32>,
  // CHECK: c = #kgen.simd<true, false>
  c = #pop.simd_floor< #kgen.simd<true, false> : !kgen.simd<2, bool>> : !kgen.simd<2, bool>,
  // CHECK: d = #pop.simd_floor<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  d = #pop.simd_floor< #kgen.unknown : !kgen.simd<4, si32>>,
  // CHECK: e = #kgen.simd<"1", "-2", "0">
  e = #pop.simd_floor< #kgen.simd<"1.9", "-1.2", "0.0"> : !kgen.simd<3, f32>> : !kgen.simd<3, f32>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 7> : !kgen.scalar<si32>
  a = #pop.simd_ceil< #kgen<simd 7> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<3, -4>
  b = #pop.simd_ceil< #kgen.simd<3, -4> : !kgen.simd<2, si32>> : !kgen.simd<2, si32>,
  // CHECK: c = #kgen.simd<true, false>
  c = #pop.simd_ceil< #kgen.simd<true, false> : !kgen.simd<2, bool>> : !kgen.simd<2, bool>,
  // CHECK: d = #pop.simd_ceil<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  d = #pop.simd_ceil< #kgen.unknown : !kgen.simd<4, si32>>,
  // CHECK: e = #kgen.simd<"2", "-1", "0">
  e = #pop.simd_ceil< #kgen.simd<"1.9", "-1.2", "0.0"> : !kgen.simd<3, f32>> : !kgen.simd<3, f32>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 7> : !kgen.scalar<si32>
  a = #pop.simd_trunc< #kgen<simd 7> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<3, -4>
  b = #pop.simd_trunc< #kgen.simd<3, -4> : !kgen.simd<2, si32>> : !kgen.simd<2, si32>,
  // CHECK: c = #kgen.simd<true, false>
  c = #pop.simd_trunc< #kgen.simd<true, false> : !kgen.simd<2, bool>> : !kgen.simd<2, bool>,
  // CHECK: d = #pop.simd_trunc<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
  d = #pop.simd_trunc< #kgen.unknown : !kgen.simd<4, si32>>,
  // CHECK: e = #kgen.simd<"1", "-1", "0">
  e = #pop.simd_trunc< #kgen.simd<"1.9", "-1.2", "0.0"> : !kgen.simd<3, f32>> : !kgen.simd<3, f32>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 4> : !kgen.scalar<si32>
  a = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<4, 4, -48, 99> : !kgen.simd<4, si16>
  b = #pop.simd_shl< #kgen.simd<1, 2, -3, 99> : !kgen.simd<4, si16>, #kgen.simd<2, 1, 4, 0> : !kgen.simd<4, si16>> : !kgen.simd<4, si16>,
  // CHECK: c = #pop.simd_shl<#kgen<simd 1>{{.*}}, #kgen<simd 17>
  c = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<si16>, #kgen<simd 17> : !kgen.scalar<si16>> : !kgen.scalar<si16>,
  // CHECK: d = #kgen<simd 32> : !kgen.scalar<si8>
  d = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<si8>, #kgen<simd 5> : !kgen.scalar<index>> : !kgen.scalar<si8>,
  // CHECK: e = #kgen<simd 64> : !kgen.scalar<index>
  e = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<index>, #kgen<simd 6> : !kgen.scalar<index>> : !kgen.scalar<index>,
  // CHECK: f = #kgen<simd 64> : !kgen.scalar<uindex>
  f = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<uindex>, #kgen<simd 6> : !kgen.scalar<uindex>> : !kgen.scalar<uindex>,
  // Index shl that does NOT fold without target: shift >= 32 is poison in 32-bit.
  // CHECK: g = #pop.simd_shl<#kgen<simd 1>{{.*}}, #kgen<simd 33>
  g = #pop.simd_shl< #kgen<simd 1> : !kgen.scalar<index>, #kgen<simd 33> : !kgen.scalar<index>> : !kgen.scalar<index>
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen<simd 0> : !kgen.scalar<si32>
  a = #pop.simd_shr< #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<si32>> : !kgen.scalar<si32>,
  // CHECK: b = #kgen.simd<0, 1, -1, 99> : !kgen.simd<4, si16>
  b = #pop.simd_shr< #kgen.simd<1, 2, -3, 99> : !kgen.simd<4, si16>, #kgen.simd<2, 1, 4, 0> : !kgen.simd<4, si16>> : !kgen.simd<4, si16>,
  // CHECK: c = #pop.simd_shr<#kgen<simd 1>{{.*}}, #kgen<simd 17>
  c = #pop.simd_shr< #kgen<simd 1> : !kgen.scalar<si16>, #kgen<simd 17> : !kgen.scalar<si16>> : !kgen.scalar<si16>,
  // CHECK: d = #kgen.simd<1, 4095> : !kgen.simd<2, ui16>
  d = #pop.simd_shr< #kgen.simd<65535, 65533> : !kgen.simd<2, ui16>, #kgen.simd<15, 4> : !kgen.simd<2, ui16>> : !kgen.simd<2, ui16>,
  // CHECK: e = #kgen.simd<1, 4095> : !kgen.simd<2, ui16>
  e = #pop.simd_shr< #kgen.simd<65535, 65533> : !kgen.simd<2, ui16>, #kgen.simd<15, 4> : !kgen.simd<2, index>> : !kgen.simd<2, ui16>,
  // CHECK: f = #kgen.simd<1, 4095> : !kgen.simd<2, uindex>
  f = #pop.simd_shr< #kgen.simd<65535, 65533> : !kgen.simd<2, uindex>, #kgen.simd<15, 4> : !kgen.simd<2, uindex>> : !kgen.simd<2, uindex>,
  // Index shr that does NOT fold without target: 32-bit and 64-bit results differ.
  // CHECK: g = #pop.simd_shr<#kgen<simd 3000000000>{{.*}}, #kgen<simd 1>
  g = #pop.simd_shr< #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 1> : !kgen.scalar<index>> : !kgen.scalar<index>
} : () -> ()


"some.op"() {
  // CHECK: a = #kgen.simd<7, 7, 1, -2147483648>
  a = #pop.simd_abs< #kgen.simd<7, -7, -1, -2147483648> : !kgen.simd<4, si32>>,
  // CHECK: b = #kgen.simd<"7", "7", "NaN", "+Inf">
  b = #pop.simd_abs< #kgen.simd<"7", "-7", "-NaN", "-inf"> : !kgen.simd<4, f32> >,
  // CHECK: c = #pop.simd_abs<#kgen.unknown :
  c = #pop.simd_abs< #kgen.unknown : !kgen.scalar<si32>>,
  // CHECK: d = #kgen.simd<true, false>
  d = #pop.simd_abs< #kgen.simd<true, false> : !kgen.simd<2, bool>>,
  // CHECK: e = #kgen.simd<1, 8, 9223090564025548800, -9223372036854775808>
  e = #pop.simd_abs<  #kgen.simd<-1, -8, 9223090564025548800, -9223372036854775808> : !kgen.simd<4, index>>,
  // CHECK: f = #pop.simd_abs<#kgen<simd 9223372036854775807>
  f = #pop.simd_abs< #kgen.simd<9223372036854775807> : !kgen.scalar<index> >
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen.simd<7, -7, -1, -2147483648
  a = #pop.simd_round< #kgen.simd<7, -7, -1, -2147483648> : !kgen.simd<4, si32>>,
  // CHECK: b = #kgen.simd<"7", "-7", "2", "1", "2", "-1", "-2", "-Inf">
  b = #pop.simd_round< #kgen.simd<"7.0", "-7.0", "1.5", "1.1", "1.7", "-1.2", "-1.7", "-inf"> : !kgen.simd<8, f32> >,
  // CHECK: c = #pop.simd_round<#kgen.unknown :
  c = #pop.simd_round< #kgen.unknown : !kgen.scalar<si32>>,
  // CHECK: d = #kgen.simd<true, false>
  d = #pop.simd_round< #kgen.simd<true, false> : !kgen.simd<2, bool>>,
  // CHECK: e = #kgen.simd<-1, -8, 9223090564025548800, -9223372036854775808>
  e = #pop.simd_round<  #kgen.simd<-1, -8, 9223090564025548800, -9223372036854775808> : !kgen.simd<4, index>>,
  // CHECK: f = #kgen<simd 9223372036854775807>
  f = #pop.simd_round< #kgen.simd<9223372036854775807> : !kgen.scalar<index> >,
  // CHECK: g = #kgen<simd "7">
  g = #pop.simd_round< #kgen<sugar alias, !kgen.scalar<f32>, #kgen.simd<"6.1">, #kgen.simd<"7.1">> >
} : () -> ()

"some.op"() {
  // CHECK: a = #kgen.simd<2, -3, -1, 0>
  a = #pop.simd_floordiv< #kgen.simd<7, 7, -1, 0> : !kgen.simd<4, si32>,
                          #kgen.simd<3, -3, 1024, -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>,
  // CHECK: b = #kgen.simd<"2", "-3">
  b = #pop.simd_floordiv< #kgen.simd<"7", "7"> : !kgen.simd<2, f32>,
                          #kgen.simd<"3", "-3"> : !kgen.simd<2, f32>> : !kgen.simd<2, f32>,
  // CHECK: c = #pop.simd_floordiv<#kgen.unknown :
  c = #pop.simd_floordiv< #kgen.unknown : !kgen.scalar<si32>, #kgen.simd<1> : !kgen.scalar<si32>>,
  // CHECK: d = #pop.simd_floordiv<#kgen<simd 1> : !kgen.scalar<si32>, #kgen.unknown
  d = #pop.simd_floordiv<  #kgen.simd<1> : !kgen.scalar<si32>, #kgen.unknown : !kgen.scalar<si32>>,
  // CHECK: e = #kgen<simd -3>
  e = #pop.simd_floordiv<  #kgen.simd<7> : !kgen.scalar<index>, #kgen.simd<-3> : !kgen.scalar<index>>,
  // CHECK: f = #pop.simd_floordiv<#kgen<simd 9223372036854775807>
  f = #pop.simd_floordiv<  #kgen.simd<9223372036854775807> : !kgen.scalar<index>, #kgen.simd<-3> : !kgen.scalar<index>>
} : () -> ()
